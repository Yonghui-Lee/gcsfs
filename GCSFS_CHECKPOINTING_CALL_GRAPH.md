# GCSFS Checkpointing Call Graphs & Architectural Flow Report

This report provides an exhaustive, production-grade architectural analysis and step-by-step call graphs for saving and loading deep learning model checkpoints to/from Google Cloud Storage (GCS) using `gcsfs`.

It covers two primary axes:
1. **The Checkpointing Paradigm**: **Monolithic Checkpointing** (`torch.save` / `torch.load`) vs. **PyTorch Distributed Checkpointing (DCP)** (`torch.distributed.checkpoint`).
2. **The Bucket Topology / Storage Path**: **Standard Non-Zonal GCS Buckets** (HTTP JSON/XML API) vs. **Zonal Hierarchical Namespace (HNS) Buckets** (High-Performance gRPC Bidirectional Streaming API).

---

## 1. Architectural Overview

The client-side and cloud-side interaction models differ significantly depending on the bucket layout and protocol under both writing and reading pathways:

```text
                                GCS STORAGE TARGETS & PATHS
                                
                    ┌─────────────────────────────────────────────────┐
                    │            gcsfs (ExtendedGcsFileSystem)        │
                    └────────────┬──────────────────────┬─────────────┘
                                 │                      │
                   (Standard GCS Bucket)          (Zonal HNS Bucket)
                                 │                      │
                                 ▼                      ▼
                           [GCSFile]               [ZonalFile]
                                 │                      │
                       (HTTP JSON/XML API)         (gRPC Stream)
                                 │                      │
                      Resumable Upload/GET        BidiWrite/Read
                                 ▼                      ▼
                            [Standard GCS]         [Zonal GCS HNS]
```

---

## 2. Monolithic Checkpointing Pipeline (Saving / `torch.save`)

Under Monolithic Checkpointing, all model parameters and optimizer momentum variables are gathered onto a single process (usually Rank 0). Rank 0 then serializes this unified dictionary of tensors sequentially and streams it as a single monolithic checkpoint file (e.g., `.ckpt` or `.pt`) directly to GCS.

In practice, monolithic checkpoint saving behaves differently based on whether it is orchestrated through PyTorch Lightning's standard callback mechanism or streamed directly to GCS via `fsspec` in a manual/custom script.

---

### Scenario A: PyTorch Lightning `ModelCheckpoint` Default Saving Pipeline

By default, PyTorch Lightning's `ModelCheckpoint` callback wraps saving in an **atomic write** workflow to protect against half-written or corrupted checkpoint files on cloud storage. This atomic write serializes the entire checkpoint payload into host RAM first before initiating any cloud filesystem connection.

#### High-Level Call Graph (Scenario A)

```text
Trainer.save_checkpoint()  [PyTorch Lightning]
   │
   ▼
ModelCheckpoint._save_checkpoint()  [PyTorch Lightning]
   │
   ▼
Trainer.save_checkpoint(filepath)  [PyTorch Lightning]
   │
   ▼
DDPStrategy.save_checkpoint(checkpoint, filepath)  [PyTorch Lightning Strategy]
   │
   ▼
TorchCheckpointIO.save_checkpoint(checkpoint, filepath)  [PyTorch Lightning CheckpointIO]
   │
   ▼
_atomic_save(checkpoint, filepath)  [lightning.fabric.utilities.cloud_io]
   │
   ├──► 1. torch.save(checkpoint, bytesbuffer)  [PyTorch Core] -> Serializes to in-memory io.BytesIO() RAM buffer
   │        │
   │        ▼
   │     torch.serialization._save(obj, zip_file)  [PyTorch Core]
   │        │
   │        ▼
   │     pickle.dump()  [Python Standard Library] (Writes entire state dictionary to BytesIO buffer)
   │
   └──► 2. fsspec.core.url_to_fs(str(filepath))  [fsspec Core] -> Resolves filesystem instance (ExtendedGcsFileSystem)
            │
            ▼
        with fs.transaction, fs.open(urlpath, "wb") as f:  [fsspec Core/gcsfs]
            │
            ├──► ExtendedGcsFileSystem._open(path, mode="wb")  [gcsfs] -> Returns ZonalFile or GCSFile
            │
            └──► f.write(bytesbuffer.getvalue())  [gcsfs] -> Writes the entire serialized checkpoint chunk to the file
```

#### Detailed Step-by-Step Call Path (Scenario A)

1. **`ModelCheckpoint._save_checkpoint(trainer, filepath)`**
   * **Location**: `lightning/pytorch/callbacks/model_checkpoint.py`
   * **Action**: Captures checkpoint triggers (such as epoch or validation step completions) and delegates saving to the main trainer.
   * **Code Invocation**: `trainer.save_checkpoint(filepath, self.save_weights_only)`

2. **`Trainer.save_checkpoint(filepath, weights_only)`**
   * **Location**: `lightning/pytorch/trainer/trainer.py`
   * **Action**: Extracts training loop, scheduler, optimizer, and model state dictionaries into a unified checkpoint structure and routes the request to the active strategy.
   * **Code Invocation**: `self.strategy.save_checkpoint(checkpoint, filepath)`

3. **`Strategy.save_checkpoint(checkpoint, filepath)`**
   * **Location**: Active training strategy (e.g., `DDPStrategy` in `lightning/pytorch/strategies/ddp.py`)
   * **Action**: Ensures only the global Rank 0 process performs the save to prevent race conditions and network write conflicts across nodes. Delegates the save operation to the strategy's `CheckpointIO` plugin.
   * **Code Invocation**: `self.checkpoint_io.save_checkpoint(checkpoint, filepath)`

4. **`TorchCheckpointIO.save_checkpoint(checkpoint, filepath)`**
   * **Location**: `lightning/plugins/io/torch_plugin.py`
   * **Action**: Serves as PyTorch Lightning's standard file-writing backend. Invokes Lightning's internal atomic-saving cloud IO module.
   * **Code Invocation**: `lightning.fabric.utilities.cloud_io._atomic_save(checkpoint, filepath)`

5. **`_atomic_save(checkpoint, filepath)`**
   * **Location**: `lightning/fabric/utilities/cloud_io.py`
   * **Action**: Implements an in-memory buffer-to-disk atomic pattern:
     * **Step 1: Serialize to RAM**: Instantiates an in-memory memory stream `bytesbuffer = io.BytesIO()`. It then calls **`torch.save(checkpoint, bytesbuffer)`**. This serializes the entire 44.87 GiB payload sequentially into RAM on Rank 0. `torch.save` delegates to `torch.serialization._save(obj, zip_file)` which uses the Python Standard Library's `pickle.dump()` to convert python objects to a sequential binary stream.
     * **Step 2: Resolve Filesystem via fsspec**: Calls **`fs, urlpath = fsspec.core.url_to_fs(str(filepath))`**. Since the path starts with `"gs://"`, `fsspec` maps the protocol and retrieves our custom **`ExtendedGcsFileSystem`** instance.
     * **Step 3: Open Remote File**: Starts an fsspec transaction and opens the file: **`with fs.transaction, fs.open(urlpath, "wb") as f:`**. This triggers `ExtendedGcsFileSystem._open()`.
     * **Step 4: Bulk Write to Storage**: Calls **`f.write(bytesbuffer.getvalue())`** to dump the entire pre-serialized checkpoint memory block in bulk writes directly to the open `ZonalFile` or `GCSFile`.

---

### Scenario B: Direct / Manual `torch.save` Streaming Pipeline

Some customized checkpointing setups or direct profiling scripts (such as our `profile_monolithic.py` script) bypass PyTorch Lightning's in-memory `BytesIO` buffer entirely to save Rank 0 RAM and avoid Out-Of-Memory (OOM) failures on massive checkpoints. Instead, they open an `fsspec` stream directly and pass it to `torch.save`, causing the pickling serialization engine to issue thousands of small write operations directly to the cloud storage target.

#### High-Level Call Graph (Scenario B)

```text
fsspec.open(filepath, "wb")  [fsspec Core]
   │
   ▼
ExtendedGcsFileSystem._open(path, mode="wb")  [gcsfs]
   │
   ├──► _sync_lookup_bucket_type(bucket)  [gcsfs Control Plane Lookup]
   │       ├── ZONAL_HIERARCHICAL ──► Return ZonalFile
   │       └── NON_HIERARCHICAL ────► Return GCSFile
   │
   ▼ (Returns file object f)
torch.save(state_dict, f)  [PyTorch Core]
   │
   ▼
torch.serialization._save(obj, zip_file)  [PyTorch Core]
   │
   ▼
pickle.dump()  [Python Standard Library] (Single-threaded recursive pickling)
   │
   ▼ (Continuous series of f.write(data) calls of varying sizes directly to fsspec stream)
   │
   ├──► Metadata Phase: Hundreds of small writes (< 150 bytes, p50=23 bytes)
   ├──► Payload Phase: Massive raw binary tensor writes (~120 MB to ~200 MB)
   │
   ▼
   ZonalFile.write(data) or GCSFile.write(data)  [gcsfs]
```

#### Detailed Step-by-Step Call Path (Scenario B)

1. **`fsspec.open(filepath, "wb")`**
   * **Location**: `fsspec/core.py`
   * **Action**: Recognizes the `"gs"` protocol, instantiates an `ExtendedGcsFileSystem` if not cached, and triggers its open interface.
   * **Code Invocation**: `ExtendedGcsFileSystem._open(path, mode="wb", ...)`

2. **`ExtendedGcsFileSystem._open(path, mode="wb", ...)`**
   * **Location**: `gcsfs/extended_gcsfs.py`
   * **Action**: Intercepts the open, splits the path to identify the target bucket, and queries the storage layout control plane:
     * Calls **`_sync_lookup_bucket_type(bucket)`**, which performs an asynchronous control plane lookup (`get_storage_layout` RPC) cached synchronously.
     * **If Zonal HNS Bucket**: Returns an instance of **`ZonalFile`** (imported from `gcsfs/zonal_file.py`).
     * **Otherwise**: Defaults to a standard **`GCSFile`** (from `gcsfs/core.py`).

3. **`torch.save(state_dict, f)`**
   * **Location**: `torch/serialization.py`
   * **Action**: Invoked with the open `ZonalFile` or `GCSFile` object passed directly as `f`. Starts the zip container assembly and invokes the serializer.
   * **Code Invocation**: `torch.serialization._save(obj, zip_file)`

4. **`pickle.dump(obj, f, protocol)`**
   * **Location**: Python Standard `pickle` module
   * **Action**: Performs a depth-first traversal of the model parameters and optimizer state dictionary, converting items sequentially to binary. As it serializes, it issues continuous, highly skewed `.write(data)` calls directly to the file object `f`:
     * **Metadata Phase**: Hundreds of tiny writes (median p50 size of **23 bytes**, p90 of **129 bytes**) representing python schema structural declarations and ZIP archive directories.
     * **Payload Phase**: Massive, multi-megabyte raw contiguous tensor arrays (often ~120 MB to ~200 MB).

---

### Internal Write Mechanics: `ZonalFile.write(data)` vs. `GCSFile.write(data)`

Once `pickle.dump()` calls `.write(data)` on the file-like stream `f`, the data path splits depending on the target bucket type:

#### Pathway A: Standard Non-Zonal GCS Bucket (`GCSFile`)
1. **`GCSFile.write(data)`** (in `gcsfs/core.py`):
   * Appends the incoming write bytes to an internal `UnclosableBytesIO()` memory buffer.
   * Increments `self.loc` by `len(data)`.
2. **Lazy Upload Initiation (`_initiate_upload()`)**:
   * On the very first write, `GCSFile` detects that no upload session exists and calls **`self._initiate_upload()`**.
   * Sends an HTTP `POST` request to GCS to initiate a Resumable Multipart Upload and retrieves a session upload token.
3. **Resumable Chunk Uploading (`_upload_chunk(final=False)`)**:
   * As `pickle` writes continue, `GCSFile` checks if the internal buffer size matches or exceeds `block_size` (the GCS chunk write boundary).
   * If yes, it calls **`self._upload_chunk(final=False)`**, which reads the chunk from `UnclosableBytesIO()` and streams it to the remote GCS URL via an HTTP `PUT` request.

#### Pathway B: Zonal HNS Bucket with Async Offloading (`ZonalFile`)
1. **`ZonalFile.write(data)`** (in `gcsfs/zonal_file.py`):
   * Checks for lazy initialization: If `self.aaow` (AsyncAppendableObjectWriter) is None, calls **`self._ensure_aaow()`** which invokes **`zb_hns_utils.init_aaow`** synchronously on the event loop to create a gRPC stream.
   * Increments `self.loc` by `len(data)`.
2. **Force-Sync Fallback check**:
   * If `GCSFS_ZONAL_FORCE_SYNC_WRITE == "true"`: Blocks the main thread and writes immediately via **`asyn.sync(self.gcsfs.loop, self.aaow.append, data)`**, paying the context-switch and network synchronization latency penalty on every tiny 23-byte metadata write.
3. **Asynchronous Write Task Scheduling**:
   * If force-sync is false, it schedules the append task asynchronously onto the `gcsfs` background asyncio event loop via **`asyncio.run_coroutine_threadsafe(self._schedule_append(data), self.gcsfs.loop)`**, returning a `Future` object immediately without blocking the main Python serialization thread.
4. **FIFO Task Chaining (`_schedule_append(data)`)**:
   * Inside `_schedule_append(data)`, it tracks tasks sequentially using `self._last_async_task`. It await-chains the previous scheduled task to guarantee strict First-In-First-Out (FIFO) sequential append ordering:
     ```python
     if previous_task:
         await previous_task # Wait for previous chunk's gRPC send to complete
     await self.aaow.append(data) # Stream current chunk over gRPC
     ```
5. **Backpressure and OOM Prevention**:
   * Appends the `Future` to `self._pending_futures`.
   * If the queue of outstanding concurrent writes matches or exceeds `self._max_pending_writes` (controlled via `GCSFS_ZONAL_MAX_PENDING_WRITES`, default `2`), it blocks the thread and waits for the oldest pending write to finish:
     ```python
     while len(self._pending_futures) >= self._max_pending_writes:
         self._pending_futures.pop(0).result() # Blocks until gRPC flush clears
     ```

---

### Finalization and Close

* **`GCSFile.close()`**:
  * Forces the remaining buffer to GCS via **`_upload_chunk(final=True)`**, locking the object.
* **`ZonalFile.close()`**:
  * Blocks and waits for all queued background async writes to finish: **`self._wait_for_pending_futures()`**.
  * Closes the underlying `AsyncAppendableObjectWriter` via **`asyn.sync(self.gcsfs.loop, zb_hns_utils.close_aaow, self.aaow, finalize_on_close=self.finalize_on_close)`**.
  * If `finalize_on_close=True`, it performs a gRPC object finalize RPC, locking the file from further appends and committing it to GCS HNS.

---

## 3. Monolithic Checkpoint Reading Pipeline (`torch.load` / Loading)

During model checkpoint restoration or validation stages, the process is inverted. Rank 0 opens the monolithic checkpoint file from cloud storage and streams it back to the host CPU memory, unpickling the structures and allocating tensor storages.

### High-Level Call Graph (Reading / Loading)

```text
Trainer.fit(ckpt_path=...)  [PyTorch Lightning]
   │
   ▼
Trainer._checkpoint_connector.resume_start(checkpoint_path)  [PyTorch Lightning]
   │
   ▼
TorchCheckpointIO.load_checkpoint(checkpoint_path)  [PyTorch Lightning CheckpointIO]
   │
   ▼
_load(checkpoint_path, map_location, weights_only)  [lightning.fabric.utilities.cloud_io]
   │
   ├──► 1. get_filesystem(checkpoint_path)  [lightning.fabric.utilities.cloud_io] -> ExtendedGcsFileSystem
   │
   ├──► 2. with fs.open(checkpoint_path, "rb") as f:  [fsspec Core/gcsfs]
   │        │
   │        ▼
   │     ExtendedGcsFileSystem._open(path, mode="rb")  [gcsfs]
   │        ├──► [Standard Bucket] ──► Return GCSFile
   │        └──► [Zonal HNS Bucket]  ──► Return ZonalFile
   │
   └──► 3. torch.load(f, map_location, weights_only)  [PyTorch Core]
            │
            ▼
         pickle.load() / Unpickler.load()  [Python Standard Library] (Recursive stream parsing)
            │
            ▼ (Continuous series of f.seek() and f.read(n) calls to retrieve targeted segments)
            │
            ├──► Standard GCS Bucket (GCSFile) ──► Range HTTP GETs (Resumable fetch)
            └──► Zonal HNS Bucket (ZonalFile)   ──► gRPC ReadObject / Concurrent Async MRDs
```

### Detailed Execution Trace & Methods

1. **`Trainer.fit(ckpt_path=...)`**
   * **Location**: `lightning/pytorch/trainer/trainer.py`
   * **Action**: Entry point for training. If a checkpoint path is supplied, it triggers the checkpoint connector to restore training states.

2. **`Trainer._checkpoint_connector.resume_start(checkpoint_path)`**
   * **Location**: `lightning/pytorch/trainer/connectors/checkpoint_connector.py`
   * **Action**: Coordinates training epoch, step, loop state, and hyperparameter restoration. Routes the path to the strategy's `CheckpointIO` plugin.

3. **`TorchCheckpointIO.load_checkpoint(checkpoint_path)`**
   * **Location**: `lightning/plugins/io/torch_plugin.py`
   * **Action**: Resolves the target filesystem and routes the binary file stream parsing to Lightning's cloud loader.
   * **Code Invocation**: `lightning.fabric.utilities.cloud_io._load(path, map_location, weights_only)`

4. **`_load(checkpoint_path, map_location, weights_only)`**
   * **Location**: `lightning/fabric/utilities/cloud_io.py`
   * **Action**: Coordinates streaming unpickling:
     * Resolves the filesystem: Calls **`fs = get_filesystem(checkpoint_path)`** (which retrieves the registered **`ExtendedGcsFileSystem`**).
     * Opens the remote path: **`with fs.open(checkpoint_path, "rb") as f:`**
     * This invokes **`ExtendedGcsFileSystem._open(path, mode="rb")`**, which queries the bucket layout control plane (`_sync_lookup_bucket_type`) and instantiates a **`ZonalFile`** (gRPC-backed) or **`GCSFile`** (HTTP-backed) in read mode.
     * Triggers PyTorch loading: **`torch.load(f, map_location=map_location, weights_only=weights_only)`**

5. **`torch.load(f)` and Pickle Stream Parsing**:
   * **Location**: `torch/serialization.py`
   * **Action**: Starts the ZIP container parsing and invokes the Python Standard `pickle.load()` / `Unpickler` engine.
   * **Execution Pattern**: Unlike sequential writes, `torch.load` reads files selectively. It reads ZIP catalog headers, seeks back and forth to locate specific metadata tables, and sequentially deserializes weights.
   * **Method Calls**: Triggers numerous **`f.seek(offset)`**, **`f.tell()`**, and **`f.read(n)`** calls directly on the open `ZonalFile` or `GCSFile` instance.

6. **Internal Read Mechanics: Fetching Bytes from GCS**:
   * **Pathway A: Standard Non-Zonal GCS Bucket (`GCSFile`)**
     * **`GCSFile.read(n)`** / **`GCSFile._fetch_range(start, end)`** (in `gcsfs/core.py`):
       * Submits synchronous or asynchronous HTTP `GET` requests to the standard GCS XML/JSON API with custom **`Range: bytes=start-end`** headers, fetching chunked blocks on-demand.
   * **Pathway B: Zonal HNS Bucket (`ZonalFile`)**
     * **`ZonalFile.read(n)`** / **`ZonalFile._fetch_range(start, end)`** (in `gcsfs/zonal_file.py`):
       * Blocks and delegates range fetches to the gRPC client: **`asyn.sync(self.gcsfs.loop, self.gr_client.read_object, bucket, key, start, length)`**.
       * For large files or sequential prefetching, it leverages **`AsyncMultiRangeDownloader`** (MRD) caching pools to fetch segments concurrently over bidirectional gRPC streams.

---

## 4. Distributed Checkpointing Pipeline (DCP with Staging)

To overcome the single-threaded CPU bottleneck of Python `pickle` and single-rank write/read constraints, Distributed Checkpointing (DCP) allows all training processes (Ranks 0 to $N$) to save and load their respective sharded state dictionaries concurrently.

Our benchmark workload utilizes a **parallel-local-saving/loading and staged GCS transfer** architecture to maximize efficiency.

---

### Saving Pipeline (DCP -> Local -> GCS)

```text
======================= STEP 1: PARALLEL LOCAL SAVE (ALL RANKS) =======================

torch.distributed.checkpoint.save()  [PyTorch Core]
   │
   ├──► SavePlanner.create_local_plan()  (Inspects local sharded tensors)
   │
   ├──► SavePlanner.create_global_plan()  (Synchronizes all ranks via all_gather)
   │
   └──► FileSystemWriter.write_data()
            │
            ▼ (Parallel Disk IO)
       Local NVMe SSD  (Saves sharded files: __0_0.distcp, __1_0.distcp, etc.)


=================== STEP 2: STAGED GCS UPLOAD (RANK 0 ONLY / DISTRIBUTED) ===================

fsspec.filesystem("gs").put_file(local_file, remote_file)  [fsspec Core]
   │
   ▼
ExtendedGcsFileSystem._put_file(lpath, rpath)  [gcsfs]
   │
   ├──► [Standard Bucket] ──► super()._put_file() ──► HTTP PUT stream (Chunked Upload)
   │
   └──► [Zonal HNS Bucket]  ──► init_aaow() ──► append_from_file() ──► gRPC Stream (BidiWriteObject)
                                    │
                                    ▼
                                close_aaow(finalize_on_close=True)  (Committed to GCS HNS)
```

#### Detailed Execution Trace & Methods (Saving)

##### Step 1: Parallel Local Saving (All Ranks Concurrently)

1. **`torch.distributed.checkpoint.save(state_dict, storage_writer)`**
   * **Location**: `torch/distributed/checkpoint/state_dict_saver.py`
   * **Action**: Main coordinator entrypoint for parallel distributed saving.

2. **`SavePlanner.create_local_plan()`**
   * **Location**: `torch/distributed/checkpoint/default_planner.py`
   * **Action**: Every process rank analyzes its model and optimizer shards, computing metadata such as tensor shapes, data types, and slice offsets.

3. **`SavePlanner.create_global_plan()`**
   * **Location**: `torch/distributed/checkpoint/default_planner.py`
   * **Action**: Executes a collective distributed `all_gather` across all participating processes to merge individual local plans. Rank 0 calculates the global coordinate map, planning exactly which tensor segments map to which `.distcp` shard files.

4. **`FileSystemWriter.write_data(plan, planner)`**
   * **Location**: `torch/distributed/checkpoint/filesystem.py`
   * **Action**: All ranks synchronously serialize their local tensor sharding states directly to raw binary arrays, skipping Python's CPU-bound `pickle` engine, and write them in parallel to individual `.distcp` files (e.g., `__0_0.distcp`, `__1_0.distcp`) on local NVMe disk.

5. **`dist.barrier()`**
   * **Action**: Suspends further execution until all ranks have safely completed local NVMe storage operations.

##### Step 2: Staged GCS Upload (Distributed/Parallelized Upload)

Once the local parallel save is verified, the checkpoint folder is synchronized to GCS. To maximize write performance and avoid Rank 0 network bottlenecks, our benchmark splits the upload task across ranks concurrently:

1. **Clean Target Directory (Rank 0 Only)**:
   * Rank 0 opens the GCS filesystem using `fs = fsspec.filesystem("gs")`.
   * Checks if the directory exists using `fs.exists(filepath)` and deletes it: **`fs.rm(filepath, recursive=True)`**.
   * Syncs across ranks using `dist.barrier()`.

2. **Local Directory Traversal**:
   * Every rank scans the local NVMe temporary folder: `local_files = os.listdir(local_dir)`.
   * Iterates through all found files (e.g., `.metadata` and sharded `.distcp` files).

3. **Shard Assignment & Selection**:
   * Each rank uploads only its own written files to prevent duplicate network traffic:
     * **Rank 0** is solely responsible for uploading the global coordinate index file: **`.metadata`**.
     * **Rank $k$** is solely responsible for uploading its specific shard files: **`__k_*.distcp`**.
   * Selected files are uploaded using **`fs.put_file(local_file_path, remote_file_path, overwrite=True)`**.

4. **`ExtendedGcsFileSystem._put_file(lpath, rpath, ...)`**:
   * **Location**: `gcsfs/extended_gcsfs.py`
   * **Action**: Intercepts the upload. If the destination is a Standard bucket, it calls `super()._put_file()` (HTTP multipart upload). If it is a Zonal HNS bucket, it performs high-speed gRPC stream writing:
     1. Calls **`await self._get_grpc_client()`** to initialize gRPC connectivity.
     2. Calls **`writer = await zb_hns_utils.init_aaow(self.grpc_client, bucket, key)`** to open an `AsyncAppendableObjectWriter` over gRPC.
     3. Opens the local file: **`with open(lpath, "rb") as f:`**
     4. Pipes the local file stream to the active gRPC stream: **`await writer.append_from_file(f, block_size=chunksize)`**. This translates directly into high-performance `BidiWriteObject` gRPC requests.
     5. Calls **`await zb_hns_utils.close_aaow(writer, finalize_on_close=True)`** to finalize and commit the object.
     6. Updates local GCS filesystem directory cache: **`await self._write_file_cache_update(rpath)`**.

5. **`dist.barrier()`**:
   * Suspends further training steps until all participating ranks have fully completed their remote GCS uploads.

---

### Loading Pipeline (GCS -> Local -> DCP)

To restore a distributed checkpoint, the process is inverted. The sharded binary files are downloaded to local storage before loading in parallel.

```text
=================== STEP 1: STAGED GCS DOWNLOAD (ALL RANKS / DISTRIBUTED) ===================

fsspec.filesystem("gs").get_file(remote_file, local_file)  [fsspec Core]
   │
   ▼
ExtendedGcsFileSystem._get_file_request(rpath, lpath)  [gcsfs]
   │
   ├──► [Standard Bucket] ──► super()._get_file_request() ──► Sequential HTTP GET Chunks
   │
   └──► [Zonal HNS Bucket]  ──► self._mrd_pool_cache.get() ──► AsyncMultiRangeDownloader (MRD)
                                                                   │
                                                                   ▼
                                                            gRPC Stream (BidiReadObject)


======================= STEP 2: PARALLEL LOCAL LOAD (ALL RANKS) =======================

torch.distributed.checkpoint.load()  [PyTorch Core]
   │
   ├──► LoadPlanner.create_local_plan()  (Maps parameters to locally requested slices)
   │
   ├──► LoadPlanner.create_global_plan()  (Coordinates slice layout globally across ranks)
   │
   └──► FileSystemReader.read_data()
            │
            ▼ (Parallel Disk Read)
       Local NVMe SSD  (Loads sharded files: __0_0.distcp, __1_0.distcp, etc.)
```

#### Detailed Execution Trace & Methods (Loading)

##### Step 1: Staged GCS Download (Distributed Parallel Download)

To load a distributed checkpoint, the directory of shard files must first be downloaded to local NVMe storage on each training rank:

1. **Verify Checkpoint Directory**:
   * Ranks access the remote GCS checkpoint directory: `fs = fsspec.filesystem("gs")`.
   * Retrieves remote folder file listing: `remote_files = fs.listdir(filepath)`.

2. **Distributed File-to-Rank Mapping**:
   * To maximize network download throughput and avoid duplicate cross-process writing, each process rank downloads a designated subset of files:
     * **Rank 0** downloads the coordinate index: **`.metadata`**.
     * **Rank $k$** downloads its assigned binary shards: **`__k_*.distcp`**.
   * Downloads are triggered via: **`fs.get_file(remote_file_path, local_file_path)`**.

3. **`ExtendedGcsFileSystem._get_file_request(rpath, lpath, ...)`**:
   * **Location**: `gcsfs/extended_gcsfs.py`
   * **Action**: Intercepts the download:
     * **If Standard GCS Bucket**: Calls `super()._get_file_request()` to download bytes sequentially over chunked standard HTTP `GET` requests.
     * **If Zonal HNS Bucket (gRPC Optimized)**:
       * Instantiates or fetches a cached Multi-Reader pool: **`mrd_pool = await self._mrd_pool_cache.get(bucket, key, generation)`**.
       * Acquires a downloader stream: **`async with mrd_pool.get_mrd() as mrd:`**
       * Streams chunks of size `chunksize` (default `128 KiB`) from the active gRPC bidirectional stream, writing them directly to the local NVMe file stream.

4. **`dist.barrier()`**:
   * Suspends execution until all ranks have successfully finalized downloading their local shards from GCS.

##### Step 2: Parallel Local Loading (All Ranks Concurrently)

1. **`torch.distributed.checkpoint.load(state_dict, storage_reader)`**
   * **Location**: `torch/distributed/checkpoint/state_dict_saver.py`
   * **Action**: Serves as the main coordinator for parallel distributed loading.
   * **Code Invocation**: `dcp.load(state_dict, storage_reader=dcp.FileSystemReader(local_dir))`

2. **`LoadPlanner.create_local_plan()`**:
   * **Location**: `torch/distributed/checkpoint/default_planner.py`
   * **Action**: Each rank scans its local model/optimizer tensor allocations and builds a map of slices it is responsible for holding.

3. **`LoadPlanner.create_global_plan()`**:
   * **Location**: `torch/distributed/checkpoint/default_planner.py`
   * **Action**: Executes a collective `all_gather` across ranks. Reads the downloaded `.metadata` file, mapping local requested slices to coordinates stored across the sharded `.distcp` files.

4. **`FileSystemReader.read_data(plan, planner)`**:
   * **Location**: `torch/distributed/checkpoint/filesystem.py`
   * **Action**: All ranks simultaneously open their locally stored shard files on NVMe SSD and read raw binary tensor blocks directly into GPU/CPU memory structures in parallel, completely avoiding any Python `pickle` bottlenecks or network desynchronizations.

---

## 5. Architectural Comparison Summary

| Metric / Feature | Monolithic Checkpointing (`torch.save` -> GCS) | Distributed Checkpointing (DCP -> Local -> GCS) |
| :--- | :--- | :--- |
| **Serialization Engine** | Python `pickle` (Single-threaded CPU bound) | PyTorch Binaries (Multi-threaded & Parallel) |
| **CPU Serialization Core(s)** | **1 CPU Core** (Rank 0 alone) | **All CPU Cores** (Parallel across ranks) |
| **I/O Bottleneck Path** | Single-Rank network stream / serial CPU unpickling | Sharded writing, parallel local IO, staged transfer |
| **Bucket Storage Format** | Single monolithic `.ckpt` / `.pt` file | Sharded directory structure with global `.metadata` index |
| **GCS Zonal HNS Protocol** | Standard `GCSFile` or `ZonalFile` write loop | `_put_file` / `_get_file_request` via direct gRPC stream |
| **Resharding Capability** | Unsupported (Requires loading on same topology) | Fully Supported (Can load checkpoint into different rank layout) |
| **Memory Buffer Overhead** | Scenario A: Doubles RAM on Rank 0 (`BytesIO` buffer) <br> Scenario B: None (direct stream, but causes slow IO context-switches without async offloading) | Extremely Low (Writes and reads local files directly as shards) |
| **Monolithic Loading Mechanics** | ZIP directory parsing issuing sequential `.seek()` and `.read()` calls directly on the fsspec handle. | N/A (Uses localized directory parallel loading) |
| **Parallel Downloading Mechanics** | N/A | Cached `AsyncMultiRangeDownloader` pools streaming concurrent chunks over gRPC to local NVMe. |
