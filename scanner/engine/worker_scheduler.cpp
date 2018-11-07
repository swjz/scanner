/* Copyright 2016 Carnegie Mellon University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "scanner/engine/worker.h"
#include "scanner/engine/evaluate_worker.h"
#include "scanner/engine/kernel_registry.h"
#include "scanner/engine/source_registry.h"
#include "scanner/engine/sink_registry.h"
#include "scanner/engine/load_worker.h"
#include "scanner/engine/runtime.h"
#include "scanner/engine/save_worker.h"
#include "scanner/engine/table_meta_cache.h"
#include "scanner/engine/python_kernel.h"
#include "scanner/engine/dag_analysis.h"
#include "scanner/util/cuda.h"
#include "scanner/util/glog.h"
#include "scanner/util/grpc.h"
#include "scanner/util/fs.h"

#include <arpa/inet.h>
#include <grpc/grpc_posix.h>
#include <grpc/support/log.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <sys/socket.h>
#include <pybind11/embed.h>

#ifdef __linux__
#include <omp.h>
#endif


// For avcodec_register_all()... should go in software video with global mutex
extern "C" {
#include "libavcodec/avcodec.h"
}

using storehouse::StoreResult;
using storehouse::WriteFile;
using storehouse::RandomReadFile;
namespace py = pybind11;

namespace scanner {
namespace internal {

namespace {
inline bool operator==(const MemoryPoolConfig& lhs,
                       const MemoryPoolConfig& rhs) {
  return (lhs.cpu().use_pool() == rhs.cpu().use_pool()) &&
         (lhs.cpu().free_space() == rhs.cpu().free_space()) &&
         (lhs.gpu().use_pool() == rhs.gpu().use_pool()) &&
         (lhs.gpu().free_space() == rhs.gpu().free_space());
}

inline bool operator!=(const MemoryPoolConfig& lhs,
                       const MemoryPoolConfig& rhs) {
  return !(lhs == rhs);
}

// This function parses EvalWorkEntry struct to protocol buffer format
void evalworkentry_to_proto(const EvalWorkEntry& input, proto::EvalWorkEntry& output,
    Profiler& profiler) {
  auto parse_start = now();

  output.set_table_id(input.table_id);
  output.set_job_index(input.job_index);
  output.set_task_index(input.task_index);
  for (const std::vector<i64>& col_row_ids : input.row_ids) {
    proto::ColumnRowIds* col_row_ids_proto = output.add_col_row_ids();
    for (const i64& row_id : col_row_ids) {
      col_row_ids_proto->add_row_ids(row_id);
    }
  }
  for (auto i = 0; i < input.columns.size(); ++i) {
    const Elements& elements = input.columns.at(i);
    const DeviceHandle& handle = input.column_handles.at(i);
    proto::Elements* elements_proto = output.add_columns();
    for (const Element& element : elements) {
      proto::Element* element_proto = elements_proto->add_element();
      element_proto->set_size(element.size);
      if (element.is_frame) {
        const Frame* frame = element.as_const_frame();
        auto create_buffer_start = now();
        std::vector<u8> buffer;
        buffer.resize(frame->size());
        memcpy_buffer(buffer.data(), CPU_DEVICE, frame->data, handle, frame->size());
        profiler.add_interval("create_buffer", create_buffer_start, now());
        element_proto->set_buffer(buffer.data(), buffer.size());
        for (i32 shape : frame->shape) {
          element_proto->add_shape(shape);
        }
        element_proto->set_type(frame->type);
      } else {
        std::vector<u8> buffer;
        buffer.resize(element.size);
        memcpy_buffer(buffer.data(), CPU_DEVICE, element.buffer, handle, element.size);
        element_proto->set_buffer(buffer.data(), buffer.size());
      }
      element_proto->set_index(element.index);
      element_proto->set_is_frame(element.is_frame);
    }
  }
  for (const DeviceHandle& column_handle : input.column_handles) {
    proto::DeviceHandle* column_handle_proto = output.add_column_handles();
    column_handle_proto->set_id(column_handle.id);
    column_handle_proto->set_type(column_handle.type);
  }
  for (const bool inplace_video_option : input.inplace_video) {
    output.add_inplace_video(inplace_video_option);
  }
  for (const ColumnType& column_type : input.column_types) {
    output.add_column_types(column_type);
  }
  output.set_needs_configure(input.needs_configure);
  output.set_needs_reset(input.needs_reset);
  output.set_last_in_io_packet(input.last_in_io_packet);
  for (const proto::VideoDescriptor::VideoCodecType& video_encoding_type : input.video_encoding_type) {
    if (video_encoding_type == proto::VideoDescriptor::RAW) {
      output.add_video_encoding_type(proto::EvalWorkEntry::RAW);
    } else if (video_encoding_type == proto::VideoDescriptor::H264) {
      output.add_video_encoding_type(proto::EvalWorkEntry::H264);
    }
  }
  output.set_first(input.first);
  output.set_last_in_task(input.last_in_task);
  for (const FrameInfo& frame_size : input.frame_sizes) {
    proto::FrameInfo* frame_size_proto = output.add_frame_sizes();
    frame_size_proto->set_type(frame_size.type);
    for (i32 shape : frame_size.shape) {
      frame_size_proto->add_shape(shape);
    }
  }
  for (const bool compressed : input.compressed) {
    output.add_compressed(compressed);
  }

  profiler.add_interval("eval_to_proto", parse_start, now());
}

// This function parses protocol buffer format to EvalWorkEntry struct
void proto_to_evalworkentry(const proto::EvalWorkEntry& input, EvalWorkEntry& output,
    Profiler& profiler, const DeviceHandle& handle) {
  auto parse_start = now();
  output.table_id = input.table_id();
  output.job_index = input.job_index();
  output.task_index = input.task_index();
  for (int i=0; i<input.col_row_ids_size(); ++i) {
    output.row_ids.emplace_back();
    std::vector<i64>& col_row_ids = output.row_ids.back();
    const proto::ColumnRowIds& col_row_ids_proto = input.col_row_ids(i);
    for (int j=0; j<col_row_ids_proto.row_ids_size(); ++j) {
      col_row_ids.push_back(col_row_ids_proto.row_ids(j));
    }
  }
  for (int i=0; i<input.column_handles_size(); ++i) {
    output.column_handles.emplace_back();
    DeviceHandle& column_handle = output.column_handles.back();
    const proto::DeviceHandle& column_handle_proto = input.column_handles(i);
    // NOTE(swjz): Since the handle of previous output might not be the real handle
    // this task uses here, we force the input handle of this task to be the real one
    column_handle.id = handle.id;
    column_handle.type = handle.type;
  }
  for (int i=0; i<input.columns_size(); ++i) {
    bool is_allocated = false;
    u8* frame_buffer = nullptr;
    output.columns.emplace_back();
    Elements& elements = output.columns.back();
//    const DeviceHandle& handle = output.column_handles.at(i);
    const proto::Elements& elements_proto = input.columns(i);
    for (int j=0; j<elements_proto.element_size(); ++j) {
      elements.emplace_back();
      Element& element = elements.back();
      auto& element_proto = elements_proto.element(j);
      if (element_proto.size() > 0) {
        if (element_proto.is_frame()) {
          FrameInfo info{};
          for (int k=0; k<element_proto.shape_size(); ++k) {
            info.shape[k] = element_proto.shape(k);
          }
          info.type = element_proto.type();

          // Here we assume that either all elements, or no element
          // in this column are frames
          if (!is_allocated) {
            is_allocated = true;
            auto allocate_start = now();
            // The frame_buffer here is essentially frame->data() rather than Element.buffer
            VLOG(1) << "Allocating buffer of size " << info.size() * elements_proto.element_size();
            frame_buffer = new_block_buffer(handle, info.size() * elements_proto.element_size(),
                                            elements_proto.element_size());
            profiler.add_interval("allocation", allocate_start, now());
          }
          u8* this_frame_buffer = frame_buffer + j * info.size();
          auto memcpy_start = now();
          memcpy_buffer(this_frame_buffer, handle, (u8*)element_proto.buffer().c_str(),
              CPU_DEVICE, info.size());
          profiler.add_interval("memcpy_frame", memcpy_start, now());
          Frame* frame = new Frame(info, this_frame_buffer);
          element.buffer = reinterpret_cast<u8*>(frame);
        } else {
          element.buffer = new_buffer(handle, element_proto.size());
          auto memcpy_start = now();
          memcpy_buffer(element.buffer, handle, (u8*)element_proto.buffer().c_str(),
              CPU_DEVICE, element_proto.size());
          profiler.add_interval("memcpy", memcpy_start, now());
        }
      }
      element.size = element_proto.size();
      element.is_frame = element_proto.is_frame();
      element.index = element_proto.index();
    }
  }
  for (int i=0; i<input.inplace_video_size(); ++i) {
    output.inplace_video.push_back(input.inplace_video(i));
  }
  for (int i=0; i<input.column_types_size(); ++i) {
    output.column_types.push_back(input.column_types(i));
  }
  output.needs_configure = input.needs_configure();
  output.needs_reset = input.needs_reset();
  output.last_in_io_packet = input.last_in_io_packet();
  for (int i=0; i<input.video_encoding_type_size(); ++i) {
    output.video_encoding_type.emplace_back();
    proto::VideoDescriptor::VideoCodecType& video_encoding_type =
        output.video_encoding_type.back();
    if (input.video_encoding_type(i) == proto::EvalWorkEntry::RAW) {
      video_encoding_type = proto::VideoDescriptor::RAW;
    } else if (input.video_encoding_type(i) == proto::EvalWorkEntry::H264) {
      video_encoding_type = proto::VideoDescriptor::H264;
    }
  }
  output.first = input.first();
  output.last_in_task = input.last_in_task();
  for (int i=0; i<input.frame_sizes_size(); ++i) {
    output.frame_sizes.emplace_back();
    FrameInfo& frame_sizes = output.frame_sizes.back();
    frame_sizes.type = input.frame_sizes(i).type();
    for (int j=0; j<input.frame_sizes(i).shape_size(); ++j) {
      frame_sizes.shape[j] = input.frame_sizes(i).shape(j);
    }
  }
  for (int i=0; i<input.compressed_size(); ++i) {
    output.compressed.push_back(input.compressed(i));
  }

  profiler.add_interval("proto_to_eval", parse_start, now());
}

void load_driver(LoadInputQueue& load_work,
                 std::vector<EvalQueue>& initial_eval_work,
                 LoadWorkerArgs args) {
  Profiler& profiler = args.profiler;
  LoadWorker worker(args);
  while (true) {
    auto idle_start = now();

    std::tuple<i32, std::deque<TaskStream>, LoadWorkEntry> entry;
    load_work.pop(entry);
    i32& output_queue_idx = std::get<0>(entry);
    auto& task_streams = std::get<1>(entry);
    LoadWorkEntry& load_work_entry = std::get<2>(entry);

    args.profiler.add_interval("idle", idle_start, now());

    if (load_work_entry.job_index() == -1) {
      break;
    }

    VLOG(2) << "Load (N/PU: " << args.node_id << "/" << args.worker_id
            << "): processing job task (" << load_work_entry.job_index() << ", "
            << load_work_entry.task_index() << ")";

    auto work_start = now();

    auto input_entry = load_work_entry;
    worker.feed(input_entry);

    while (true) {
      EvalWorkEntry output_entry;
      i32 io_packet_size = args.io_packet_size;
      if (worker.yield(io_packet_size, output_entry)) {
        auto& work_entry = output_entry;
        work_entry.first = !task_streams.empty();
        work_entry.last_in_task = worker.done();
        initial_eval_work[output_queue_idx].push(
            std::make_tuple(task_streams, work_entry));
        // We use the task streams being empty to indicate that this is
        // a new task, so clear it here to show that this is from the same task
        task_streams.clear();
      } else {
        break;
      }
    }
    profiler.add_interval("task", work_start, now());
    VLOG(2) << "Load (N/PU: " << args.node_id << "/" << args.worker_id
            << "): finished job task (" << load_work_entry.job_index() << ", "
            << load_work_entry.task_index() << "), pushed to worker "
            << output_queue_idx;
  }
  VLOG(1) << "Load (N/PU: " << args.node_id << "/" << args.worker_id
          << "): thread finished";
}

std::map<int, std::mutex> no_pipelining_locks;
std::map<int, std::condition_variable> no_pipelining_cvars;
std::map<int, bool> no_pipelining_conditions;

void pre_evaluate_driver(EvalQueue& input_work, EvalQueue& output_work,
                         PreEvaluateWorkerArgs args) {
  Profiler& profiler = args.profiler;
  PreEvaluateWorker worker(args);
  // We sort inputs into task work queues to ensure we process them
  // sequentially
  std::map<std::tuple<i32, i32>,
           Queue<std::tuple<std::deque<TaskStream>, EvalWorkEntry>>>
      task_work_queue;
  i32 work_packet_size = args.work_packet_size;

  std::tuple<i32, i32> active_job_task = std::make_tuple(-1, -1);
  while (true) {
    auto idle_start = now();

    // If we have no work at all or we do not have work for our current task..
    if (task_work_queue.empty() ||
        (std::get<0>(active_job_task) != -1 &&
         task_work_queue.at(active_job_task).size() <= 0)) {
      std::tuple<std::deque<TaskStream>, EvalWorkEntry> entry;
      input_work.pop(entry);


      auto& task_streams = std::get<0>(entry);
      EvalWorkEntry& work_entry = std::get<1>(entry);
      VLOG(1) << "Pre-evaluate (N/KI: " << args.node_id << "/" << args.worker_id
              << "): got work " << work_entry.job_index << " " << work_entry.task_index;
      if (work_entry.job_index == -1) {
        break;
      }

      VLOG(1) << "Pre-evaluate (N/KI: " << args.node_id << "/" << args.worker_id
              << "): "
              << "received job task " << work_entry.job_index << ", "
              << work_entry.task_index;

      task_work_queue[std::make_tuple(work_entry.job_index,
                                      work_entry.task_index)]
          .push(entry);
    }

    args.profiler.add_interval("idle", idle_start, now());

    if (std::get<0>(active_job_task) == -1) {
      // Choose the next task to work on
      active_job_task = task_work_queue.begin()->first;
    }

    // Wait until we have the next io item for the current task
    if (task_work_queue.at(active_job_task).size() <= 0) {
      std::this_thread::yield();
      continue;
    }

    // Grab next entry for active task
    std::tuple<std::deque<TaskStream>, EvalWorkEntry> entry;
    task_work_queue.at(active_job_task).pop(entry);

    auto& task_streams = std::get<0>(entry);
    EvalWorkEntry& work_entry = std::get<1>(entry);

    VLOG(1) << "Pre-evaluate (N/KI: " << args.node_id << "/" << args.worker_id
            << "): "
            << "processing job task " << work_entry.job_index << ", "
            << work_entry.task_index;

    auto work_start = now();

    i32 total_rows = 0;
    for (size_t i = 0; i < work_entry.row_ids.size(); ++i) {
      total_rows = std::max(total_rows, (i32)work_entry.row_ids[i].size());
    }

    bool first = work_entry.first;
    bool last = work_entry.last_in_task;

    auto input_entry = work_entry;
    worker.feed(input_entry, first);
    i32 rows_used = 0;
    while (rows_used < total_rows) {
      EvalWorkEntry output_entry;
      if (!worker.yield(work_packet_size, output_entry)) {
        break;
      }

      if (std::getenv("NO_PIPELINING")) {
        no_pipelining_conditions[args.worker_id] = true;
      }

      if (first) {
        output_work.push(std::make_tuple(task_streams, output_entry));
        first = false;
      } else {
        output_work.push(
            std::make_tuple(std::deque<TaskStream>(), output_entry));
      }

      if (std::getenv("NO_PIPELINING")) {
        std::unique_lock<std::mutex> lk(no_pipelining_locks[args.worker_id]);
        no_pipelining_cvars[args.worker_id].wait(lk, [&] {
          return !no_pipelining_conditions[args.worker_id];
        });
      }
      rows_used += work_packet_size;
    }

    if (last) {
      task_work_queue.erase(active_job_task);
      active_job_task = std::make_tuple(-1, -1);
    }

    profiler.add_interval("task", work_start, now());
  }

  VLOG(1) << "Pre-evaluate (N/PU: " << args.node_id << "/" << args.worker_id
          << "): thread finished ";
}

void evaluate_driver(EvalQueue& input_work, EvalQueue& output_work,
                     EvaluateWorkerArgs args) {
  Profiler& profiler = args.profiler;
  EvaluateWorker worker(args);
  while (true) {
    auto idle_pull_start = now();

    std::tuple<std::deque<TaskStream>, EvalWorkEntry> entry;
    input_work.pop(entry);

    auto& task_streams = std::get<0>(entry);
    EvalWorkEntry& work_entry = std::get<1>(entry);

    args.profiler.add_interval("idle_pull", idle_pull_start, now());

    if (work_entry.job_index == -1) {
      break;
    }

    VLOG(1) << "Evaluate (N/KI/G: " << args.node_id << "/" << args.ki << "/"
            << args.kg << "): processing job task " << work_entry.job_index
            << ", " << work_entry.task_index;

    auto work_start = now();

    if (task_streams.size() > 0) {
      // Start of a new task. Tell kernels what outputs they should produce.
      std::vector<TaskStream> streams;
      for (i32 i = 0; i < args.arg_group.kernel_factories.size(); ++i) {
        assert(!task_streams.empty());
        streams.push_back(task_streams.front());
        task_streams.pop_front();
      }
      worker.new_task(work_entry.job_index, work_entry.task_index, streams);
    }

    i32 work_packet_size = 0;
    for (size_t i = 0; i < work_entry.columns.size(); ++i) {
      work_packet_size =
          std::max(work_packet_size, (i32)work_entry.columns[i].size());
    }

    auto input_entry = work_entry;
    worker.feed(input_entry);
    EvalWorkEntry output_entry;
    bool result = worker.yield(work_packet_size, output_entry);
    (void)result;
    assert(result);

    profiler.add_interval("task", work_start, now());

    auto idle_push_start = now();
    output_work.push(std::make_tuple(task_streams, output_entry));
    args.profiler.add_interval("idle_push", idle_push_start, now());

  }
  VLOG(1) << "Evaluate (N/KI: " << args.node_id << "/" << args.ki
          << "): thread finished";
}

void post_evaluate_driver(EvalQueue& input_work, OutputEvalQueue& output_work,
                          PostEvaluateWorkerArgs args) {
  Profiler& profiler = args.profiler;
  PostEvaluateWorker worker(args);
  while (true) {
    auto idle_start = now();

    std::tuple<std::deque<TaskStream>, EvalWorkEntry> entry;
    input_work.pop(entry);
    EvalWorkEntry& work_entry = std::get<1>(entry);

    args.profiler.add_interval("idle", idle_start, now());

    if (work_entry.job_index == -1) {
      break;
    }

    VLOG(1) << "Post-evaluate (N/PU: " << args.node_id << "/" << args.id
            << "): processing task " << work_entry.job_index << ", "
            << work_entry.task_index;

    auto work_start = now();

    auto input_entry = work_entry;
    worker.feed(input_entry);
    EvalWorkEntry output_entry;
    bool result = worker.yield(output_entry);
    profiler.add_interval("task", work_start, now());

    if (result) {
      VLOG(1) << "Post-evaluate (N/PU: " << args.node_id << "/" << args.id
              << "): pushing task " << work_entry.job_index << ", "
              << work_entry.task_index;

      output_entry.last_in_task = work_entry.last_in_task;
      output_work.push(std::make_tuple(args.id, output_entry));
    }

    if (std::getenv("NO_PIPELINING")) {
      {
          std::unique_lock<std::mutex> lk(no_pipelining_locks[args.id]);
          no_pipelining_conditions[args.id] = false;
      }
      no_pipelining_cvars[args.id].notify_one();
    }
  }

  VLOG(1) << "Post-evaluate (N/PU: " << args.node_id << "/" << args.id
          << "): thread finished ";
}

void save_coordinator(OutputEvalQueue& eval_work,
                      std::vector<SaveInputQueue>& save_work) {
  i32 num_save_workers = save_work.size();
  std::map<std::tuple<i32, i32>, i32> task_to_worker_mapping;
  i32 last_worker_assigned = 0;
  while (true) {
    auto idle_start = now();

    std::tuple<i32, EvalWorkEntry> entry;
    eval_work.pop(entry);
    EvalWorkEntry& work_entry = std::get<1>(entry);

    //args.profiler.add_interval("idle", idle_start, now());

    if (work_entry.job_index == -1) {
      break;
    }

    VLOG(1) << "Save Coordinator: "
            << "processing job task (" << work_entry.job_index << ", "
            << work_entry.task_index << ")";

    auto job_task_id =
        std::make_tuple(work_entry.job_index, work_entry.task_index);
    if (task_to_worker_mapping.count(job_task_id) == 0) {
      // Assign worker to this task
      task_to_worker_mapping[job_task_id] =
          last_worker_assigned++ % num_save_workers;
    }

    i32 assigned_worker = task_to_worker_mapping.at(job_task_id);
    save_work[assigned_worker].push(entry);

    if (work_entry.last_in_task) {
      task_to_worker_mapping.erase(job_task_id);
    }

    VLOG(1) << "Save Coordinator: "
            << "finished job task (" << work_entry.job_index << ", "
            << work_entry.task_index << ")";

  }
}

void save_driver(SaveInputQueue& save_work,
                 SaveOutputQueue& output_work,
                 SaveWorkerArgs args) {
  Profiler& profiler = args.profiler;
  std::map<std::tuple<i32, i32>, std::unique_ptr<SaveWorker>> workers;
  while (true) {
    auto idle_start = now();

    std::tuple<i32, EvalWorkEntry> entry;
    save_work.pop(entry);

    i32 pipeline_instance = std::get<0>(entry);
    EvalWorkEntry& work_entry = std::get<1>(entry);

    args.profiler.add_interval("idle", idle_start, now());

    if (work_entry.job_index == -1) {
      break;
    }

    VLOG(1) << "Save (N/KI: " << args.node_id << "/" << args.worker_id
            << "): processing job task (" << work_entry.job_index << ", "
            << work_entry.task_index << ")";

    auto work_start = now();

    // Check if we have a worker for this task
    auto job_task_id =
        std::make_tuple(work_entry.job_index, work_entry.task_index);
    if (workers.count(job_task_id) == 0) {
      SaveWorker* worker = new SaveWorker(args);
      worker->new_task(work_entry.job_index, work_entry.task_index,
                       work_entry.table_id, work_entry.column_types);
      workers[job_task_id].reset(worker);
    }

    auto& worker = workers.at(job_task_id);

    auto input_entry = work_entry;
    worker->feed(input_entry);

    VLOG(1) << "Save (N/KI: " << args.node_id << "/" << args.worker_id
            << "): finished task (" << work_entry.job_index << ", "
            << work_entry.task_index << ")";

    args.profiler.add_interval("task", work_start, now());

    if (work_entry.last_in_task) {
      auto finished_start = now();
      worker->finished();
      args.profiler.add_interval("finished", finished_start, now());

      auto workpush_start = now();
      output_work.enqueue(std::make_tuple(
          pipeline_instance, work_entry.job_index, work_entry.task_index));
      args.profiler.add_interval("workpush", workpush_start, now());
      auto save_worker_delete_start = now();
      workers.erase(job_task_id);
      args.profiler.add_interval("save_worker_delete", save_worker_delete_start,
                                 now());
    }
  }

  VLOG(1) << "Save (N/KI: " << args.node_id << "/" << args.worker_id
          << "): thread finished ";
}
}

WorkerImpl::WorkerImpl(DatabaseParameters& db_params,
                       std::string master_address, std::string worker_port)
  : watchdog_awake_(true),
    db_params_(db_params),
    state_(State::INITIALIZING),
    master_address_(master_address),
    worker_port_(worker_port) {
  init_glog("scanner_worker");

  LOG(INFO) << "Creating worker";

  {
    // HACK(apoms): to fix this issue: https://github.com/pybind/pybind11/issues/1364
    pybind11::get_shared_data("");
  }

  set_database_path(db_params.db_path);

  avcodec_register_all();
#ifdef DEBUG
  // Stop SIG36 from grpc when debugging
  grpc_use_signal(-1);
#endif
  // google::protobuf::io::CodedInputStream::SetTotalBytesLimit(67108864 * 4,
  //                                                            67108864 * 2);

  LOG(INFO) << "Create master stub";
  master_ = proto::Master::NewStub(
      grpc::CreateChannel(master_address, grpc::InsecureChannelCredentials()));
  LOG(INFO) << "Finish master stub";

  storage_ =
      storehouse::StorageBackend::make_from_config(db_params_.storage_config);

  // Processes jobs in the background
  start_job_processor();
  LOG(INFO) << "Worker created.";

  std::stringstream resource_ss;
  resource_ss << "Resources: ";
  resource_ss << "CPUs: " << db_params_.num_cpus << ", ";
  resource_ss << "Save: " << db_params_.num_save_workers << ", ";
  resource_ss << "Load: " << db_params_.num_load_workers << ", ";
  resource_ss << "GPU ids: ";
  for (i32 g : db_params_.gpu_ids) {
    resource_ss << g << " ";
  }
  resource_ss << ".";
  VLOG(2) << resource_ss.rdbuf();
}

WorkerImpl::~WorkerImpl() {
  State state = state_.get();
  bool was_initializing = state == State::INITIALIZING;
  state_.set(State::SHUTTING_DOWN);

  // Master is dead if we failed during initialization
  if (!was_initializing) {
    try_unregister();
  }

  trigger_shutdown_.set();

  stop_job_processor();

  if (watchdog_thread_.joinable()) {
    watchdog_thread_.join();
  }
  delete storage_;
  if (memory_pool_initialized_) {
    destroy_memory_allocators();
  }
}

grpc::Status WorkerImpl::NewJob(grpc::ServerContext* context,
                                const proto::BulkJobParameters* job_params,
                                proto::Result* job_result) {
  LOG(INFO) << "Worker " << node_id_ << " received NewJob";
  // Ensure that only one job is running at a time and that the worker
  // is in idle mode before transitioning to job start
  State state = state_.get();
  bool ready = false;
  while (!ready) {
    switch (state) {
      case RUNNING_JOB: {
        RESULT_ERROR(job_result, "This worker is already running a job!");
        return grpc::Status::OK;
      }
      case SHUTTING_DOWN: {
        RESULT_ERROR(job_result, "This worker is preparing to shutdown!");
        return grpc::Status::OK;
      }
      case INITIALIZING: {
        state_.wait_for_change(INITIALIZING);
        break;
      }
      case IDLE: {
        if (state_.test_and_set(state, RUNNING_JOB)) {
          ready = true;
          break;
        }
      }
    }
    state = state_.get();
  }

  job_result->set_success(true);
  set_database_path(db_params_.db_path);

  job_params_.Clear();
  job_params_.MergeFrom(*job_params);
  {
    std::unique_lock<std::mutex> lock(finished_mutex_);
    finished_ = false;
  }
  finished_cv_.notify_all();

  {
    std::unique_lock<std::mutex> lock(active_mutex_);
    active_bulk_job_ = true;
    active_bulk_job_id_ = job_params->bulk_job_id();
  }
  active_cv_.notify_all();

  return grpc::Status::OK;
}

grpc::Status WorkerImpl::Shutdown(grpc::ServerContext* context,
                                  const proto::Empty* empty, Result* result) {
  State state = state_.get();
  switch (state) {
    case RUNNING_JOB: {
      // trigger_shutdown will inform job to stop working
      break;
    }
    case SHUTTING_DOWN: {
      // Already shutting down
      result->set_success(true);
      return grpc::Status::OK;
    }
    case INITIALIZING: {
      break;
    }
    case IDLE: {
      break;
    }
  }
  state_.set(SHUTTING_DOWN);
  try_unregister();
  // Inform watchdog that we are done for
  trigger_shutdown_.set();
  active_cv_.notify_all();
  result->set_success(true);
  return grpc::Status::OK;
}

grpc::Status WorkerImpl::Ping(grpc::ServerContext* context,
                              const proto::Empty* empty1,
                              proto::PingReply* reply) {
  reply->set_node_id(node_id_);
  watchdog_awake_ = true;
  return grpc::Status::OK;
}

void WorkerImpl::start_watchdog(grpc::Server* server, bool enable_timeout,
                                i32 timeout_ms) {
  watchdog_thread_ = std::thread([this, server, enable_timeout, timeout_ms]() {
    double time_since_check = 0;
    // Wait until shutdown is triggered or watchdog isn't woken up
    if (!enable_timeout) {
      trigger_shutdown_.wait();
    }
    while (!trigger_shutdown_.raised()) {
      auto sleep_start = now();
      trigger_shutdown_.wait_for(timeout_ms);
      time_since_check += nano_since(sleep_start) / 1e6;
      if (time_since_check > timeout_ms) {
        if (!watchdog_awake_) {
          // Watchdog not woken, time to bail out
          LOG(ERROR) << "Worker did not receive heartbeat in " << timeout_ms
                     << "ms. Shutting down.";
          trigger_shutdown_.set();
        }
        watchdog_awake_ = false;
        time_since_check = 0;
      }
    }
    // Shutdown self
    server->Shutdown();
  });
}

Result WorkerImpl::register_with_master() {
  assert(state_.get() == State::INITIALIZING);

  LOG(INFO) << "Worker try to register with master";

  proto::WorkerParams worker_info;
  worker_info.set_port(worker_port_);
  proto::MachineParameters* params = worker_info.mutable_params();
  params->set_num_cpus(db_params_.num_cpus);
  params->set_num_load_workers(db_params_.num_cpus);
  params->set_num_save_workers(db_params_.num_cpus);
  for (i32 gpu_id : db_params_.gpu_ids) {
    params->add_gpu_ids(gpu_id);
  }

  proto::Registration registration;
  grpc::Status status;
  GRPC_BACKOFF(master_->RegisterWorker(&ctx, worker_info, &registration),
               status);
  if (!status.ok()) {
    Result result;
    result.set_success(false);
    LOG(WARNING)
      << "Worker could not contact master server at " << master_address_ << " ("
      << status.error_code() << "): " << status.error_message();
    return result;
  }

  node_id_ = registration.node_id();

  LOG(INFO) << "Worker registered with master with id " << node_id_;

  state_.set(State::IDLE);

  Result result;
  result.set_success(true);
  return result;
}

void WorkerImpl::try_unregister() {
  if (state_.get() != State::INITIALIZING && !unregistered_.test_and_set()) {
    proto::UnregisterWorkerRequest node_info;
    node_info.set_node_id(node_id_);

    proto::Empty em;
    grpc::Status status;
    GRPC_BACKOFF_D(master_->UnregisterWorker(&ctx, node_info, &em), status, 15);
    if (!status.ok()) {
      LOG(WARNING) << "Worker could not unregister from master server "
                   << "(" << status.error_code()
                   << "): " << status.error_message();
      return;
    }
    LOG(INFO) << "Worker unregistered from master server.";
  }
}

void WorkerImpl::load_op(const proto::OpPath* op_path) {
  std::string so_path = op_path->path();
  LOG(INFO) << "Worker " << node_id_ << " loading Op library: " << so_path;

  auto l = std::string("__stdlib").size();
  if (so_path.substr(0, l) == "__stdlib") {
    so_path = db_params_.python_dir + "/lib/libscanner_stdlib" + so_path.substr(l);
  }

  void* handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  LOG_IF(FATAL, handle == nullptr)
      << "dlopen of " << so_path << " failed: " << dlerror();
}

void WorkerImpl::register_op(const proto::OpRegistration* op_registration) {
  const std::string& name = op_registration->name();
  const bool variadic_inputs = op_registration->variadic_inputs();
  std::vector<Column> input_columns;
  size_t i = 0;
  for (auto& c : op_registration->input_columns()) {
    Column col;
    col.set_id(i++);
    col.set_name(c.name());
    col.set_type(c.type());
    input_columns.push_back(col);
  }
  std::vector<Column> output_columns;
  i = 0;
  for (auto& c : op_registration->output_columns()) {
    Column col;
    col.set_id(i++);
    col.set_name(c.name());
    col.set_type(c.type());
    output_columns.push_back(col);
  }
  bool can_stencil = op_registration->can_stencil();
  std::vector<i32> stencil(op_registration->preferred_stencil().begin(),
                           op_registration->preferred_stencil().end());
  if (stencil.empty()) {
    stencil = {0};
  }
  bool has_bounded_state = op_registration->has_bounded_state();
  i32 warmup = op_registration->warmup();
  bool has_unbounded_state = op_registration->has_unbounded_state();
  OpInfo* info = new OpInfo(name, variadic_inputs, input_columns,
                            output_columns, can_stencil, stencil,
                            has_bounded_state, warmup, has_unbounded_state, "");
  OpRegistry* registry = get_op_registry();
  registry->add_op(name, info);
  LOG(INFO) << "Worker " << node_id_ << " registering Op: " << name;
}

void WorkerImpl::register_python_kernel(const proto::PythonKernelRegistration* python_kernel) {
  const std::string& op_name = python_kernel->op_name();
  DeviceType device_type = python_kernel->device_type();
  const std::string& kernel_code = python_kernel->kernel_code();
  const std::string& pickled_config = python_kernel->pickled_config();
  const int batch_size = python_kernel->batch_size();

  // Set all input and output columns to be CPU
  std::map<std::string, DeviceType> input_devices;
  std::map<std::string, DeviceType> output_devices;
  bool can_batch = batch_size > 1;
  bool can_stencil;
  {
    OpRegistry* registry = get_op_registry();
    OpInfo* info = registry->get_op_info(op_name);
    if (info->variadic_inputs()) {
      assert(device_type != DeviceType::GPU);
    } else {
      for (const auto& in_col : info->input_columns()) {
        input_devices[in_col.name()] = DeviceType::CPU;
      }
    }
    for (const auto& out_col : info->output_columns()) {
      output_devices[out_col.name()] = DeviceType::CPU;
    }
    can_stencil = info->can_stencil();
  }

  // Create a kernel builder function
  auto constructor = [op_name, kernel_code, pickled_config,
                      can_batch, can_stencil](const KernelConfig& config) {
      return new PythonKernel(config, op_name, kernel_code, pickled_config,
                              can_batch, can_stencil);
  };

  // Create a new kernel factory
  KernelFactory* factory =
      new KernelFactory(op_name, device_type, 1, input_devices, output_devices,
                        can_batch, batch_size, constructor);
  // Register the kernel
  KernelRegistry* registry = get_kernel_registry();
  registry->add_kernel(op_name, factory);
  LOG(INFO) << "Worker " << node_id_ << " registering Python Kernel: " << op_name;
}

void WorkerImpl::start_job_processor() {
  job_processor_thread_ = std::thread([this]() {
    while (!trigger_shutdown_.raised()) {
      // Wait on not finished
      {
        std::unique_lock<std::mutex> lock(active_mutex_);
        active_cv_.wait(lock, [this] {
          return active_bulk_job_ || trigger_shutdown_.raised();
        });
      }
      if (trigger_shutdown_.raised()) break;
      // Start processing job
      bool result = process_job(&job_params_, &job_result_);
      if (!result) {
        try_unregister();
        unregistered_.clear();
        state_.set(INITIALIZING);
        register_with_master();
      } else {
        // Set to idle if we finished without a shutdown
        state_.test_and_set(RUNNING_JOB, IDLE);
      }
    }
  });
}

void WorkerImpl::stop_job_processor() {
  // Wake up job processor
  {
    std::unique_lock<std::mutex> lock(active_mutex_);
    active_bulk_job_ = true;
  }
  active_cv_.notify_all();
  if (job_processor_thread_.joinable()) {
    job_processor_thread_.join();
  }
}

bool WorkerImpl::process_job(const proto::BulkJobParameters* job_params,
                             proto::Result* job_result) {
  timepoint_t base_time(std::chrono::nanoseconds(job_params->base_time()));

  Profiler profiler(base_time);
  job_result->set_success(true);

  // Set profiler level
  PROFILER_LEVEL = static_cast<ProfilerLevel>(job_params->profiler_level());

  auto setup_ops_start = now();
  // Load Ops, register Ops, and register python kernels before running jobs
  {
    proto::Empty empty;
    proto::ListLoadedOpsReply reply;
    grpc::Status status;
    GRPC_BACKOFF(master_->ListLoadedOps(&ctx, empty, &reply), status);
    if (!status.ok()) {
      RESULT_ERROR(job_result, "Worker %d could not ListLoadedOps from master",
                   node_id_);
    }

    for (auto& reg : reply.registrations()) {
      if (so_paths_.count(reg.path()) == 0) {
        load_op(&reg);
      }
    }
  }

  {
    proto::Empty empty;
    proto::ListRegisteredOpsReply reply;
    grpc::Status status;
    GRPC_BACKOFF(master_->ListRegisteredOps(&ctx, empty, &reply), status);
    if (!status.ok()) {
      RESULT_ERROR(job_result,
                   "Worker %d could not ListRegisteredOps from master",
                   node_id_);
    }

    OpRegistry* registry = get_op_registry();
    for (auto& reg : reply.registrations()) {
      if (!registry->has_op(reg.name())) {
        register_op(&reg);
      }
    }
  }

  {
    proto::Empty empty;
    proto::ListRegisteredPythonKernelsReply reply;
    grpc::Status status;
    GRPC_BACKOFF(master_->ListRegisteredPythonKernels(&ctx, empty, &reply),
                 status);
    if (!status.ok()) {
      RESULT_ERROR(
          job_result,
          "Worker %d could not ListRegisteredPythonKernels from master",
          node_id_);
    }

    KernelRegistry* registry = get_kernel_registry();
    for (auto& reg : reply.registrations()) {
      if (!registry->has_kernel(reg.op_name(), reg.device_type())) {
        register_python_kernel(&reg);
      }
    }
  }
  profiler.add_interval("process_job:ops_setup", setup_ops_start, now());

  auto finished_fn = [&]() {
    if (!trigger_shutdown_.raised()) {
      proto::FinishedJobRequest params;
      params.set_node_id(node_id_);
      params.set_bulk_job_id(active_bulk_job_id_);
      params.mutable_result()->CopyFrom(job_result_);
      proto::Empty empty;
      grpc::Status status;
      GRPC_BACKOFF(master_->FinishedJob(&ctx, params, &empty), status);
      LOG_IF(FATAL, !status.ok())
          << "Worker could not send FinishedJob to master ("
          << status.error_code() << "): " << status.error_message() << ". "
          << "Failing since the master could hang if it sees the worker is "
          << "still alive but has not finished its job.";
    }

    {
      std::unique_lock<std::mutex> lock(finished_mutex_);
      finished_ = true;
    }
    finished_cv_.notify_all();
    {
      std::unique_lock<std::mutex> lock(finished_mutex_);
      active_bulk_job_ = false;
    }
    active_cv_.notify_all();
  };

  if (!job_result->success()) {
    finished_fn();
    return false;
  }

  set_database_path(db_params_.db_path);

  // Controls if work should be distributed roundrobin or dynamically
  bool distribute_work_dynamically = true;

  const i32 work_packet_size = job_params->work_packet_size();
  const i32 io_packet_size = job_params->io_packet_size() != -1
                                 ? job_params->io_packet_size()
                                 : work_packet_size;
  i32 warmup_size = 0;

  OpRegistry* op_registry = get_op_registry();
  std::vector<proto::Job> jobs(job_params->jobs().begin(),
                               job_params->jobs().end());
  std::vector<proto::Op> ops(job_params->ops().begin(),
                             job_params->ops().end());


  auto meta_cache_start = now();
  // Setup table metadata cache for use in other operations
  DatabaseMetadata meta(job_params->db_meta());
  TableMetaCache table_meta(storage_, meta);
  profiler.add_interval("process_job:cache_table_metadata", meta_cache_start, now());
  // Perform analysis on the graph to determine:
  //
  // - populate_analysis_info: Analayze the graph to build lookup structures to
  //   the different types of ops in the graph (input, sampling, slice/unslice,
  //   output) and their parameters (warmup, batch, stencil)
  //
  // - determine_input_rows_to_slices: When the worker receives a set of outputs
  //   to process, it needs to know which slice those outputs are a part of in
  //   order to determine which parameters to bind to the graph (remember that
  //   each subsequence in a slice can have different arguments bound to the
  //   ops in the graph).
  //
  // - remap_input_op_edges: Remap multiple inputs to a single input op
  //
  // - perform_liveness_analysis: When to retire elements (liveness analysis)
  auto dag_analysis_start = now();
  DAGAnalysisInfo analysis_results;
  populate_analysis_info(ops, analysis_results);
  // Need slice input rows to know which slice we are in
  determine_input_rows_to_slices(meta, table_meta, jobs, ops, analysis_results, db_params_.storage_config);
  remap_input_op_edges(ops, analysis_results);
  // Analyze op DAG to determine what inputs need to be pipped along
  // and when intermediates can be retired -- essentially liveness analysis
  perform_liveness_analysis(ops, analysis_results);

  // The live columns at each op index
  std::vector<std::vector<std::tuple<i32, std::string>>>& live_columns =
      analysis_results.live_columns;
  // The columns to remove for the current kernel
  std::vector<std::vector<i32>> dead_columns =
      analysis_results.dead_columns;
  // Outputs from the current kernel that are not used
  std::vector<std::vector<i32>> unused_outputs =
      analysis_results.unused_outputs;
  // Indices in the live columns list that are the inputs to the current
  // kernel. Starts from the second evalutor (index 1)
  std::vector<std::vector<i32>> column_mapping =
      analysis_results.column_mapping;


  // Read final output columns for use in post-evaluate worker
  // (needed for determining column types)
  std::vector<Column> final_output_columns(
      job_params->output_columns().begin(),
      job_params->output_columns().end());
  // Read compression options from job_params
  std::vector<ColumnCompressionOptions> final_compression_options;
  for (auto& opts : job_params->compression()) {
    ColumnCompressionOptions o;
    o.codec = opts.codec();
    for (auto& kv : opts.options()) {
      o.options[kv.first] = kv.second;
    }
    final_compression_options.push_back(o);
  }
  assert(final_output_columns.size() == final_compression_options.size());
  profiler.add_interval("process_job:dag_analysis", dag_analysis_start, now());

  // Setup kernel factories and the kernel configs that will be used
  // to instantiate instances of the op pipeline
  auto pipeline_setup_start = now();
  KernelRegistry* kernel_registry = get_kernel_registry();
  std::vector<KernelFactory*> kernel_factories;
  std::vector<KernelConfig> kernel_configs;
  i32 num_cpus = db_params_.num_cpus;
  assert(num_cpus > 0);

  i32 total_gpus = db_params_.gpu_ids.size();
  i32 num_gpus = db_params_.gpu_ids.size();
  // Should have at least one gpu if there are gpus
  assert(db_params_.gpu_ids.size() == 0 || num_gpus > 0);
  std::vector<i32> gpu_ids;
  {
    i32 start_idx = 0;
    for (i32 i = 0; i < num_gpus; ++i) {
      gpu_ids.push_back(db_params_.gpu_ids[(start_idx + i) % total_gpus]);
    }
  }

  // Setup op args that will be passed to Ops when processing
  // the stream for that job
  // op_idx -> job idx -> slice -> args
  std::map<i64, std::vector<std::vector<std::vector<u8>>>> op_args;
  for (const auto& job : jobs) {
    for (auto& oa : job.op_args()) {
      auto& op_job_args = op_args[oa.op_index()];
      op_job_args.emplace_back();
      auto& sargs = op_job_args.back();
      // TODO(apoms): use real op_idx when we support multiple outputs
      // sargs[so.op_index()] =
      for (auto op_args : oa.op_args()) {
        sargs.emplace_back(std::vector<u8>(op_args.begin(), op_args.end()));
      }
    }
  }

  // TODO(swjz): Worker should only instantiate specific kernel
  // Go through the vector of Ops, and for each Op which represents a
  // non-special Op (i.e. has a kernel implementation) get the factory
  // for constructing instances of that op, and create the config object
  // that is used when instantiating that Op
  for (size_t i = 0; i < ops.size(); ++i) {
    auto& op = ops.at(i);
    const std::string& name = op.name();
    if (op.is_source() || op.is_sink() || is_builtin_op(name)) {
      kernel_factories.push_back(nullptr);
      kernel_configs.emplace_back();
      continue;
    }
    OpInfo* op_info = op_registry->get_op_info(name);

    DeviceType requested_device_type = op.device_type();
    if (requested_device_type == DeviceType::GPU && num_gpus == 0) {
      RESULT_ERROR(job_result,
                   "Scanner is configured with zero available GPUs but a GPU "
                   "op was requested! Please configure Scanner to have "
                   "at least one GPU using the `gpu_ids` config option.");
      finished_fn();
      return false;
    }

    // Make sure that there exists a kernel for this Op type which
    // is implemented using the requested device type.
    if (!kernel_registry->has_kernel(name, requested_device_type)) {
      RESULT_ERROR(
          job_result,
          "Requested an instance of op %s with device type %s, but no kernel "
          "exists for that configuration.",
          op.name().c_str(),
          (requested_device_type == DeviceType::CPU ? "CPU" : "GPU"));
      finished_fn();
      return false;
    }

    KernelFactory* kernel_factory =
        kernel_registry->get_kernel(name, requested_device_type);
    kernel_factories.push_back(kernel_factory);

    // Construct the kernel config (which is passed to the kernel when it is
    // constructed). This needs to read the Op DAG to get the serialized
    // arguments that the user specified when declaring the Op in the Python
    // interface. See KernelConfig class in scanner/api/kernel.h.
    KernelConfig kernel_config;
    kernel_config.node_id = node_id_;
    kernel_config.args =
        std::vector<u8>(op.kernel_args().begin(), op.kernel_args().end());
    const std::vector<Column>& output_columns = op_info->output_columns();
    for (auto& col : output_columns) {
      kernel_config.output_columns.push_back(col.name());
      kernel_config.output_column_types.push_back(col.type());
    }

    // Tell kernel what its inputs are from the Op DAG
    // (for variadic inputs)
    auto& input_columns = op_info->input_columns();
    for (int i = 0; i < op.inputs().size(); ++i) {
      auto input = op.inputs(i);
      kernel_config.input_columns.push_back(input.column());
      if (input_columns.size() == 0) {
        // We can have 0 columns in op info if variadic arguments
        kernel_config.input_column_types.push_back(ColumnType::Other);
      } else {
        kernel_config.input_column_types.push_back(input_columns[i].type());
      }
    }

    kernel_configs.push_back(kernel_config);
  }

  // Figure out op input domain size for handling boundary restriction during
  // stencil
  // Op -> job -> slice -> rows
  std::vector<std::vector<std::vector<i64>>> op_input_domain_size(ops.size());
  for (size_t i = 0; i < ops.size(); ++i) {
    // Grab one of the inputs to this op to figure out this ops input domain
    i64 input_op_idx = i;
    if (!ops.at(i).is_source()) {
      input_op_idx = ops.at(i).inputs(0).op_index();
    }
    for (const auto& op_total_rows : analysis_results.total_rows_per_op) {
      const auto& op_slice_rows = op_total_rows.at(input_op_idx);
      op_input_domain_size.at(i).push_back(op_slice_rows);
    }
  }

  // Break up kernels into groups that run on the same device
  std::vector<OpArgGroup> groups;
  if (!kernel_factories.empty()) {
    auto source_registry = get_source_registry();

    bool first_op = true;
    DeviceType last_device_type;
    groups.emplace_back();
    for (size_t i = 1; i < kernel_factories.size() - 1; ++i) {
      KernelFactory* factory = kernel_factories[i];
      // Factory is nullptr when we are on a builtin op
      if (factory != nullptr && first_op) {
        last_device_type = factory->get_device_type();
        first_op = false;
      }
      // NOTE(swjz): Each kernel is divided into one single group
      if (factory != nullptr) {
        last_device_type = factory->get_device_type();
      }
      groups.emplace_back();
      auto& op_group = groups.back().op_names;
      auto& op_source = groups.back().is_source;
      auto& op_sampling = groups.back().sampling_args;
      auto& group = groups.back().kernel_factories;
      auto& op_input = groups.back().op_input_domain_size;
      auto& oargs = groups.back().op_args;
      auto& lc = groups.back().live_columns;
      auto& dc = groups.back().dead_columns;
      auto& uo = groups.back().unused_outputs;
      auto& cm = groups.back().column_mapping;
      auto& st = groups.back().kernel_stencils;
      auto& bt = groups.back().kernel_batch_sizes;
      const std::string& op_name = ops.at(i).name();
      op_group.push_back(op_name);
      if (source_registry->has_source(op_name)) {
        op_source.push_back(true);
      } else {
        op_source.push_back(false);
      }
      if (analysis_results.slice_ops.count(i) > 0) {
        i64 local_op_idx = group.size();
        // Set sampling args
        auto& slice_outputs_per_job =
            groups.back().slice_output_rows[local_op_idx];
        for (auto& job_slice_outputs : analysis_results.slice_output_rows) {
          auto& slice_groups = job_slice_outputs.at(i);
          slice_outputs_per_job.push_back(slice_groups);
        }
        auto& slice_inputs_per_job =
            groups.back().slice_input_rows[local_op_idx];
        for (auto& job_slice_inputs : analysis_results.slice_input_rows) {
          auto& slice_groups = job_slice_inputs.at(i);
          slice_inputs_per_job.push_back(slice_groups);
        }
      }
      if (analysis_results.unslice_ops.count(i) > 0) {
        i64 local_op_idx = group.size();
        // Set sampling args
        auto& unslice_inputs_per_job =
            groups.back().unslice_input_rows[local_op_idx];
        for (auto& job_unslice_inputs : analysis_results.unslice_input_rows) {
          auto& slice_groups = job_unslice_inputs.at(i);
          unslice_inputs_per_job.push_back(slice_groups);
        }
      }
      if (analysis_results.sampling_ops.count(i) > 0 ||
          analysis_results.slice_ops.count(i) > 0) {
        i64 local_op_idx = group.size();
        // Set sampling args
        auto& sampling_args_per_job = groups.back().sampling_args[local_op_idx];
        for (auto& job : jobs) {
          for (auto& saa : job.sampling_args_assignment()) {
            if (saa.op_index() == i) {
              sampling_args_per_job.emplace_back(
                  saa.sampling_args().begin(),
                  saa.sampling_args().end());
              break;
            }
          }
        }
        assert(sampling_args_per_job.size() == jobs.size());
      }
      i64 local_op_idx = group.size();
      group.push_back(std::make_tuple(factory, kernel_configs[i]));
      op_input[local_op_idx] = op_input_domain_size[i];
      oargs[local_op_idx] = op_args[i];
      lc.push_back(live_columns[i]);
      dc.push_back(dead_columns[i]);
      uo.push_back(unused_outputs[i]);
      cm.push_back(column_mapping[i]);
      st.push_back(analysis_results.stencils[i]);
      bt.push_back(analysis_results.batch_sizes[i]);
    }
  }

  i32 num_kernel_groups = static_cast<i32>(groups.size());
  assert(num_kernel_groups > 0);  // is this actually necessary?

  // NOTE(swjz): I set the pipeline_instances_per_node = 1 here for now
  i32 pipeline_instances_per_node = 1;

  LOG(INFO) << "Initial pipeline instances per node: " << pipeline_instances_per_node;
  // If pipline_instances_per_node is -1, we set a smart default. Currently, we
  // calculate the maximum possible kernel instances without oversubscribing
  // any part of the pipeline, either CPU or GPU.
  bool has_gpu_kernel = false;
  if (pipeline_instances_per_node == -1) {
    pipeline_instances_per_node = std::numeric_limits<i32>::max();
    for (i32 kg = 0; kg < num_kernel_groups; ++kg) {
      auto& group = groups[kg].kernel_factories;
      for (i32 k = 0; k < group.size(); ++k) {
        // Skip builtin ops
        if (std::get<0>(group[k]) == nullptr) {
          continue;
        }
        KernelFactory* factory = std::get<0>(group[k]);
        DeviceType device_type = factory->get_device_type();
        i32 max_devices = factory->get_max_devices();

        i32 kg_ppn;
        if (max_devices == Kernel::UnlimitedDevices) {
          kg_ppn = 1;
        } else {
          kg_ppn = device_type == DeviceType::CPU
                       ? db_params_.num_cpus / max_devices
                       : (i32)num_gpus / max_devices;
        }
        LOG(INFO) << "Kernel Group " << k
                  << " Pipeline instances per node: " << kg_ppn;
        pipeline_instances_per_node = std::min(pipeline_instances_per_node, kg_ppn);

        if (device_type == DeviceType::GPU) {
          has_gpu_kernel = true;
        }
      }
    }
    if (pipeline_instances_per_node == std::numeric_limits<i32>::max()) {
      pipeline_instances_per_node = 1;
    }
  }

  LOG(INFO) << "Pipeline instances per node: " << pipeline_instances_per_node;

  if (pipeline_instances_per_node <= 0) {
    RESULT_ERROR(job_result,
                 "BulkJobParameters.pipeline_instances_per_node (%d) must be -1 "
                 "for auto-default or greater than 0 for manual configuration.",
                 pipeline_instances_per_node);
    finished_fn();
    return false;
  }

  // Setup source factories and source configs that will be used
  // to instantiate load worker instances
  std::vector<SourceFactory*> source_factories;
  std::vector<SourceConfig> source_configs;
  {
    auto registry = get_source_registry();
    auto input_remap = analysis_results.input_ops_to_first_op_columns;
    size_t source_ops_size = input_remap.size();
    source_factories.resize(source_ops_size);
    source_configs.resize(source_ops_size);
    for (auto kv : input_remap) {
      i32 op_idx;
      i32 col_idx;
      std::tie(op_idx, col_idx) = kv;

      auto& op = ops.at(op_idx);
      auto source_factory = registry->get_source(op.name());
      source_factories[col_idx] = source_factory;

      auto& out_cols = source_factory->output_columns();
      SourceConfig config;
      config.storage_config = db_params_.storage_config;
      for (auto& col : out_cols) {
        config.output_columns.push_back(col.name());
        config.output_column_types.push_back(col.type());
      }
      config.args =
          std::vector<u8>(op.kernel_args().begin(), op.kernel_args().end());
      config.node_id = node_id_;

      source_configs[col_idx] = config;
    }
  }

  // Setup sink factories and sink configs that will be used
  // to instantiate save worker instances
  std::vector<SinkFactory*> sink_factories;
  std::vector<SinkConfig> sink_configs;
  {
    auto registry = get_sink_registry();
    sink_factories.resize(1);
    sink_configs.resize(1);

    auto& output_op = ops.back();
    auto sink_factory = registry->get_sink(output_op.name());
    sink_factories[0] = sink_factory;

    auto& in_cols = sink_factory->input_columns();
    SinkConfig config;
    config.storage_config = db_params_.storage_config;
    for (auto& col : in_cols) {
      config.input_columns.push_back(col.name());
      config.input_column_types.push_back(col.type());
    }
    config.args =
        std::vector<u8>(output_op.kernel_args().begin(),
                        output_op.kernel_args().end());
    config.node_id = node_id_;

    sink_configs[0] = config;
  }

  // Setup sink args that will be passed to sinks when processing
  // the stream for that job
  // job idx -> op_idx -> args
  std::vector<std::map<i32, std::vector<u8>>> sink_args;
  for (const auto& job : jobs) {
    sink_args.emplace_back();
    auto& sargs = sink_args.back();
    for (auto& so : job.outputs()) {
      // TODO(apoms): use real op_idx when we support multiple outputs
      //sargs[so.op_index()] =
      sargs[0] =
          std::vector<u8>(so.args().begin(), so.args().end());
    }
  }
  profiler.add_interval("process_job:pipeline_setup", pipeline_setup_start, now());

  auto memory_pool_start = now();
  // Set up memory pool if different than previous memory pool
  if (!memory_pool_initialized_ ||
      job_params->memory_pool_config() != cached_memory_pool_config_) {
    if (db_params_.num_cpus < pipeline_instances_per_node &&
        job_params->memory_pool_config().cpu().use_pool()) {
      RESULT_ERROR(job_result,
                   "Cannot oversubscribe CPUs and also use CPU memory pool");
      finished_fn();
      return false;
    }
    if (db_params_.gpu_ids.size() < pipeline_instances_per_node &&
        job_params->memory_pool_config().gpu().use_pool()) {
      RESULT_ERROR(job_result,
                   "Cannot oversubscribe GPUs and also use GPU memory pool");
      finished_fn();
      return false;
    }
    if (memory_pool_initialized_) {
      destroy_memory_allocators();
    }
    init_memory_allocators(job_params->memory_pool_config(), gpu_ids);
    cached_memory_pool_config_ = job_params->memory_pool_config();
    memory_pool_initialized_ = true;
  }
  profiler.add_interval("process_job:create_memory_pool", memory_pool_start, now());


#ifdef __linux__
  omp_set_num_threads(std::thread::hardware_concurrency());
#endif

  auto pipeline_create_start = now();
  // Setup shared resources for distributing work to processing threads
  i64 accepted_tasks = 0;
  LoadInputQueue load_work;
  std::vector<EvalQueue> initial_eval_work(pipeline_instances_per_node);
  std::vector<std::vector<EvalQueue>> eval_work(pipeline_instances_per_node);
  OutputEvalQueue output_eval_work(pipeline_instances_per_node);
  std::vector<SaveInputQueue> save_work(db_params_.num_save_workers);
  SaveOutputQueue retired_tasks;

  // TODO(swjz): Only source Op should do this.
  // Setup load workers
  i32 num_load_workers = db_params_.num_load_workers;
  std::vector<Profiler> load_thread_profilers;
  std::vector<proto::Result> load_results(num_load_workers);
  for (i32 i = 0; i < num_load_workers; ++i) {
    load_thread_profilers.emplace_back(Profiler(base_time));
  }
  std::vector<std::thread> load_threads;
//  for (i32 i = 0; i < num_load_workers; ++i) {
//    LoadWorkerArgs args{
//        // Uniform arguments
//        node_id_, table_meta,
//        // Per worker arguments
//        i, db_params_.storage_config, std::ref(load_thread_profilers[i]),
//        std::ref(load_results[i]), io_packet_size, work_packet_size,
//        source_factories, source_configs};
//
//    load_threads.emplace_back(load_driver, std::ref(load_work),
//                              std::ref(initial_eval_work), args);
//  }

  // Setup evaluate workers
  std::vector<std::vector<Profiler>> eval_profilers(
      pipeline_instances_per_node);
  std::vector<std::vector<proto::Result>> eval_results(
      pipeline_instances_per_node);
  std::vector<proto::Result> pre_eval_results(pipeline_instances_per_node);
  for (auto& result : pre_eval_results) {
    result.set_success(true);
  }

  std::vector<std::tuple<EvalQueue*, EvalQueue*>> pre_eval_queues;
  std::vector<PreEvaluateWorkerArgs> pre_eval_args;
  std::vector<std::vector<std::tuple<EvalQueue*, EvalQueue*>>> eval_queues(
      pipeline_instances_per_node);
  std::vector<std::vector<EvaluateWorkerArgs>> eval_args(
      pipeline_instances_per_node);
  std::vector<std::tuple<EvalQueue*, OutputEvalQueue*>> post_eval_queues;
  std::vector<PostEvaluateWorkerArgs> post_eval_args;

  i32 next_cpu_num = 0;
  i32 next_gpu_idx = 0;
  std::mutex startup_lock;
  std::condition_variable startup_cv;
  i32 startup_count = 0;
  i32 eval_total = 0;
  for (i32 ki = 0; ki < pipeline_instances_per_node; ++ki) {
    auto& work_queues = eval_work[ki];
    std::vector<Profiler>& eval_thread_profilers = eval_profilers[ki];
    std::vector<proto::Result>& results = eval_results[ki];
    work_queues.resize(num_kernel_groups - 1 + 2);  // +2 for pre/post
    results.resize(num_kernel_groups);
    for (auto& result : results) {
      result.set_success(true);
    }
    for (i32 i = 0; i < num_kernel_groups + 2; ++i) {
      eval_thread_profilers.push_back(Profiler(base_time));
    }

    // Evaluate worker
    DeviceHandle first_kernel_type;
    for (i32 kg = 0; kg < num_kernel_groups; ++kg) {
      auto& group = groups[kg].kernel_factories;
      std::vector<EvaluateWorkerArgs>& thread_args = eval_args[ki];
      std::vector<std::tuple<EvalQueue*, EvalQueue*>>& thread_qs =
          eval_queues[ki];
      // HACK(apoms): we assume all ops in a kernel group use the
      //   same number of devices for now.
      // for (size_t i = 0; i < group.size(); ++i) {
      KernelFactory* factory = nullptr;
      for (size_t i = 0; i < group.size(); ++i) {
        if (std::get<0>(group[i]) != nullptr) {
          factory = std::get<0>(group[i]) ;
        }
      }
      DeviceType device_type = DeviceType::CPU;
      i32 max_devices = 1;
      // Factory should only be null if we only have builtin ops
      if (factory != nullptr) {
        device_type = factory->get_device_type();
        max_devices = factory->get_max_devices();
        if (max_devices == Kernel::UnlimitedDevices) {
          max_devices = 1;
        }
      }
      if (device_type == DeviceType::CPU) {
        for (i32 i = 0; i < max_devices; ++i) {
          i32 device_id = 0;
          next_cpu_num++ % num_cpus;
          for (size_t i = 0; i < group.size(); ++i) {
            KernelConfig& config = std::get<1>(group[i]);
            config.devices.clear();
            config.devices.push_back({device_type, device_id});
          }
        }
      } else {
        for (i32 i = 0; i < max_devices; ++i) {
          i32 device_id = gpu_ids[next_gpu_idx++ % num_gpus];
          for (size_t i = 0; i < group.size(); ++i) {
            KernelConfig& config = std::get<1>(group[i]);
            config.devices.clear();
            config.devices.push_back({device_type, device_id});
          }
        }
      }
      // Get the device handle for the first kernel in the pipeline
      if (kg == 0) {
        if (group.size() == 0) {
          first_kernel_type = CPU_DEVICE;
        } else {
          first_kernel_type = std::get<1>(group[0]).devices[0];
        }
      }

      // Input work queue
      EvalQueue* input_work_queue = &work_queues[kg];
      // Create new queue for output, reuse previous queue as input
      EvalQueue* output_work_queue = &work_queues[kg + 1];
      // Create eval thread for passing data through neural net
      thread_qs.push_back(
          std::make_tuple(input_work_queue, output_work_queue));
      thread_args.emplace_back(EvaluateWorkerArgs{
          // Uniform arguments
          node_id_, startup_lock, startup_cv, startup_count,

          // Per worker arguments
          ki, kg, groups[kg], job_params->boundary_condition(),
          eval_thread_profilers[kg + 1], results[kg]});
      eval_total += 1;
    }
    // Pre evaluate worker
    {
      EvalQueue* input_work_queue;
      if (distribute_work_dynamically) {
        input_work_queue = &initial_eval_work[ki];
      } else {
        input_work_queue = &initial_eval_work[0];
      }
      EvalQueue* output_work_queue =
          &work_queues[0];
      assert(groups.size() > 0);
      pre_eval_queues.push_back(
          std::make_tuple(input_work_queue, output_work_queue));
      DeviceHandle decoder_type = std::getenv("FORCE_CPU_DECODE")
        ? CPU_DEVICE
        : first_kernel_type;
      pre_eval_args.emplace_back(PreEvaluateWorkerArgs{
          // Uniform arguments
          node_id_, num_cpus,
          std::max(1, num_cpus / pipeline_instances_per_node),
          job_params->work_packet_size(),

          // Per worker arguments
          ki, decoder_type, eval_thread_profilers.front(),
          std::ref(pre_eval_results[ki])
      });
    }

    // Post evaluate worker
    {
      auto& output_op = ops.at(ops.size() - 1);
      std::vector<std::string> column_names;
      for (auto& op_input : output_op.inputs()) {
        column_names.push_back(op_input.column());
      }

      EvalQueue* input_work_queue = &work_queues.back();
      OutputEvalQueue* output_work_queue = &output_eval_work;
      post_eval_queues.push_back(
          std::make_tuple(input_work_queue, output_work_queue));
      post_eval_args.emplace_back(PostEvaluateWorkerArgs{
          // Uniform arguments
          node_id_,

          // Per worker arguments
          ki, eval_thread_profilers.back(), column_mapping.back(),
          final_output_columns, final_compression_options,
      });
    }
  }

  // Launch eval worker threads
  std::vector<std::thread> pre_eval_threads;
  std::vector<std::vector<std::thread>> eval_threads;
  std::vector<std::thread> post_eval_threads;
//  for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
//    // Pre thread
//    pre_eval_threads.emplace_back(
//        pre_evaluate_driver, std::ref(*std::get<0>(pre_eval_queues[pu])),
//        std::ref(*std::get<1>(pre_eval_queues[pu])), pre_eval_args[pu]);
//    // Op threads
//    eval_threads.emplace_back();
//    std::vector<std::thread>& threads = eval_threads.back();
//    for (i32 kg = 0; kg < num_kernel_groups; ++kg) {
//      threads.emplace_back(
//          evaluate_driver, std::ref(*std::get<0>(eval_queues[pu][kg])),
//          std::ref(*std::get<1>(eval_queues[pu][kg])), eval_args[pu][kg]);
//    }
//    // Post threads
//    post_eval_threads.emplace_back(
//        post_evaluate_driver, std::ref(*std::get<0>(post_eval_queues[pu])),
//        std::ref(*std::get<1>(post_eval_queues[pu])), post_eval_args[pu]);
//  }

  // TODO(swjz): Only sink Op should do this.
  // Setup save coordinator
//  std::thread save_coordinator_thread(
//      save_coordinator, std::ref(output_eval_work), std::ref(save_work));

  // Setup save workers
  i32 num_save_workers = db_params_.num_save_workers;
  std::vector<Profiler> save_thread_profilers;
  // TODO: check load and save worker results
  std::vector<proto::Result> save_results(num_save_workers);
  for (i32 i = 0; i < num_save_workers; ++i) {
    save_thread_profilers.emplace_back(Profiler(base_time));
  }
  std::vector<std::thread> save_threads;
//  for (i32 i = 0; i < num_save_workers; ++i) {
//    SaveWorkerArgs args{// Uniform arguments
//                        node_id_,
//                        std::ref(sink_args),
//
//                        // Per worker arguments
//                        i, db_params_.storage_config,
//                        sink_factories, sink_configs,
//                        std::ref(save_thread_profilers[i]),
//                        std::ref(save_results[i])};
//
//    save_threads.emplace_back(save_driver, std::ref(save_work[i]),
//                              std::ref(retired_tasks), args);
//  }

  profiler.add_interval("process_job:create_pipelines", pipeline_create_start, now());

  if (job_params->profiling()) {
    auto wait_for_others_start = now();
    // Wait until all evaluate workers have started up
    std::unique_lock<std::mutex> lk(startup_lock);
    startup_cv.wait(lk, [&] {
      return eval_total == startup_count;
    });
    profiler.add_interval("process_job:wait_for_pipelines", wait_for_others_start, now());
  }

  timepoint_t start_time = now();

  // Monitor amount of work left and request more when running low
  // Round robin work
  std::vector<i64> allocated_work_to_queues(pipeline_instances_per_node);
  std::vector<i64> retired_work_for_queues(pipeline_instances_per_node);
  bool finished = false;
  // This keeps track of the last time we received a "wait_for_work" message
  // from the master. If less than 1 second have passed since this message, we
  // shouldn't ask the master for more work to avoid overloading it.
  const int MILLISECONDS_TO_WAIT_ALPHA = 20;
  const int MILLISECONDS_TO_WAIT_BETA = 1000;
  const int MILLISECONDS_TO_WAIT_RAMP = 15000;
  auto last_wait_for_work_time = now() - std::chrono::milliseconds(MILLISECONDS_TO_WAIT_ALPHA);
  while (true) {
    if (trigger_shutdown_.raised()) {
      // Abandon ship!
      LOG(INFO) << "Worker " << node_id_ << " received shutdown while in NewJob";
      RESULT_ERROR(job_result, "Worker %d shutdown while processing NewJob",
                   node_id_);
      break;
    }
    if (!job_result->success()) {
      LOG(INFO) << "Worker " << node_id_ << " in error, stopping.";
      break;
    }
    auto retire_start = now();
    // We batch up retired tasks to avoid sync overhead
    std::vector<std::tuple<i32, i64, i64>> batched_retired_tasks;
    {
      // This timed wait effectively rate limits this loop to execute once per
      // millisecond. This solves a problem where this thread would busywait and
      // decrease performance in compute-bound apps.
      std::tuple<i32, i64, i64> task_retired;
      if (retired_tasks.wait_dequeue_timed(task_retired, 1000 /* 1 ms */)) {
        batched_retired_tasks.push_back(task_retired);
      }
    }
    while (retired_tasks.size() > 0) {
      // Pull retired tasks
      std::tuple<i32, i64, i64> task_retired;
      retired_tasks.pop(task_retired);
      batched_retired_tasks.push_back(task_retired);
    }
    if (!batched_retired_tasks.empty()) {
      // Make sure the retired tasks were flushed to disk before confirming
      std::fflush(NULL);
      sync();
    }
    for (std::tuple<i32, i64, i64>& task_retired : batched_retired_tasks) {
      // Inform master that this task was finished
      proto::FinishedWorkRequest params;

      params.set_node_id(node_id_);
      params.set_bulk_job_id(active_bulk_job_id_);
      params.set_job_id(std::get<1>(task_retired));
      params.set_task_id(std::get<2>(task_retired));

      proto::Empty empty;
      grpc::Status status;
      GRPC_BACKOFF(master_->FinishedWork(&ctx, params, &empty), status);

      // Update how much is in each pipeline instances work queue
      retired_work_for_queues[std::get<0>(task_retired)] += 1;

      if (!status.ok()) {
        RESULT_ERROR(job_result,
                     "Worker %d could not tell finished work to master",
                     node_id_);
        break;
      }
    }
    scheduler_profiler.add_interval("retire", retire_start, now());
    i64 total_tasks_processed = 0;
    for (i64 t : retired_work_for_queues) {
      total_tasks_processed += t;
    }
    if (finished) {
      if (total_tasks_processed == accepted_tasks) {
        break;
      } else {
        std::this_thread::yield();
        continue;
      }
    }
    // If local amount of work is less than the amount of work we want
    // queued up, then ask the master for more work.
    i32 milliseconds_since_start = ms_since(start_time);
    i32 milliseconds_to_wait = std::max(
        MILLISECONDS_TO_WAIT_ALPHA,
        std::min(
            MILLISECONDS_TO_WAIT_BETA,
            (MILLISECONDS_TO_WAIT_BETA - MILLISECONDS_TO_WAIT_ALPHA) *
                    (milliseconds_since_start / MILLISECONDS_TO_WAIT_RAMP) +
                MILLISECONDS_TO_WAIT_ALPHA));
    i32 local_work = accepted_tasks - total_tasks_processed;
    if (local_work <
            pipeline_instances_per_node * job_params->tasks_in_queue_per_pu() &&
        ms_since(last_wait_for_work_time) > milliseconds_to_wait) {
      proto::NextWorkRequest node_info;
      node_info.set_node_id(node_id_);
      node_info.set_bulk_job_id(active_bulk_job_id_);

      proto::NextWorkReply new_work;
      grpc::Status status;
      auto next_work = now();
      GRPC_BACKOFF(master_->NextWork(&ctx, node_info, &new_work), status);
      scheduler_profiler.add_interval("nextwork", next_work, now());
      if (!status.ok()) {
        RESULT_ERROR(job_result,
                     "Worker %d could not get next work from master", node_id_);
        break;
      }

      if (new_work.wait_for_work()) {
        // Waiting for more work
        VLOG(1) << "Node " << node_id_ << " received wait for work signal.";
        last_wait_for_work_time = now();
      }
      else if (new_work.no_more_work()) {
        // No more work left
        VLOG(1) << "Node " << node_id_ << " received done signal.";
        finished = true;
      } else {
        // Perform analysis on load work entry to determine upstream
        // requirements and when to discard elements.
        // NOTE(swjz): Set task sizes for all ops to io_packet_size for now
        std::map<i64, i64> task_size_per_op;
        for (i64 i=0; i<ops.size(); i++) {
          task_size_per_op[i] = io_packet_size;
        }

        i64 table_id = new_work.table_id();
        i64 job_id = new_work.job_index();
        i64 task_id = new_work.task_index();
        i32 bulk_job_id = meta.get_bulk_job_id(job_params->job_name());
        LoadWorkEntry stenciled_entry;

        auto dep_analysis_start = now();
        // All we need from this is task_stream_map. Might also use grpc to get from master
        derive_stencil_requirements_master(
            meta, table_meta, jobs.at(new_work.job_index()), ops,
            analysis_results, job_params->boundary_condition(), job_id,
            std::vector<i64>(new_work.output_rows().begin(),
                             new_work.output_rows().end()),
            stenciled_entry, task_stream,
            db_params_.storage_config);

        // Determine which pipeline instance to allocate to
        i32 target_work_queue = -1;
        i32 min_work = std::numeric_limits<i32>::max();
        for (int i = 0; i < pipeline_instances_per_node; ++i) {
          i64 outstanding_work =
              allocated_work_to_queues[i] - retired_work_for_queues[i];
          if (outstanding_work < min_work) {
            min_work = outstanding_work;
            target_work_queue = i;
          }
        }

        DeviceHandle device_handle;
        if (ops.at(task_stream_map[task_id].op_idx).device_type() == proto::CPU) {
          device_handle = CPU_DEVICE;
        } else if (ops.at(task_stream_map[task_id].op_idx).device_type() == proto::GPU) {
          // FIXME: Support only one GPU for now
          device_handle = {DeviceType::GPU, 0};
        }

        // load_driver:
        if (task_stream_map[task_id].op_idx == 0) {
          // FIXME(swjz): The row_start here is actually how many rows are loaded by
          // other tasks before this one. If the rows are not continuous or not starting
          // from zero, this code fails because it assumes that the number wanted is
          // the first row id in its list.
          i64 row_start = task_stream_map[task_id].valid_output_rows.at(0);
          i64 row_end = row_start + task_stream_map[task_id].valid_output_rows.size();

          proto::Result load_worker_result;
          LoadWorkerArgs load_worker_args {
              // Uniform arguments
              node_id_, table_meta,
              // Per worker arguments
              0, db_params_.storage_config, std::ref(load_thread_profilers[0]),
              load_worker_result, io_packet_size, work_packet_size,
              source_factories, source_configs
          };
          LoadWorker load_worker(load_worker_args);
          load_worker.feed(stenciled_entry);
          i32 output_queue_idx = 0;
          EvalWorkEntry load_output_entry;
          load_worker.yield(task_size_per_op[0], load_output_entry, row_start, row_end);
          load_output_entry.first = true;
          load_output_entry.last_in_task = true;

          i32 work_packet_size = INT_MAX; // Ignore work packet size for now
          PreEvaluateWorkerArgs pre_evaluate_worker_args {
              // Uniform arguments
              node_id_, num_cpus, num_cpus,
              job_params->work_packet_size(),

              // Per worker arguments
              0, CPU_DEVICE, eval_profilers[0][0],  // Use CPU to decode for now
          };

          auto pre_start = now();
          PreEvaluateWorker pre_evaluate_worker(pre_evaluate_worker_args);
          EvalWorkEntry pre_evaluate_input_entry = load_output_entry;
          pre_evaluate_worker.feed(pre_evaluate_input_entry, pre_evaluate_input_entry.first);
          EvalWorkEntry pre_evaluate_output_entry;
          pre_evaluate_worker.yield(task_size_per_op[0], pre_evaluate_output_entry);
          scheduler_profiler.add_interval("pre", pre_start, now());

          std::string eval_map_file_name = eval_map_path(bulk_job_id, task_id);
          std::unique_ptr<WriteFile> eval_map_output;
          BACKOFF_FAIL(
              make_unique_write_file(storage_, eval_map_file_name, eval_map_output),
              "while trying to make write file for " + eval_map_file_name);

          proto::EvalWorkEntry pre_output_proto;
          auto serialize_start = now();
          evalworkentry_to_proto(pre_evaluate_output_entry, pre_output_proto,
              scheduler_profiler);
          scheduler_profiler.add_interval("serialize", serialize_start, now());
          size_t serialized_size = pre_output_proto.ByteSizeLong();

          std::vector<u8> eval_map_bytes;
          auto allocate_start = now();
          eval_map_bytes.resize(serialized_size);
          scheduler_profiler.add_interval("alloc", allocate_start, now());
          serialize_start = now();
          pre_output_proto.SerializeToArray(eval_map_bytes.data(), serialized_size);
          scheduler_profiler.add_interval("serialize", serialize_start, now());

          auto write_start = now();
          s_write(eval_map_output.get(), eval_map_bytes.data(), serialized_size);
          scheduler_profiler.add_interval("write", write_start, now());

          // Since it has been serialized, we delete it from memory
          for (auto& column : pre_evaluate_output_entry.columns) {
            for (auto& element : column) {
              delete_element(device_handle, element);
            }
          }
        }
        else if (ops.at(task_stream_map[task_id].op_idx).is_source()) {
          // this source op was remapped. do nothing.
        }
        // If sink Op, post evaluate & save work
        else if (ops.at(task_stream_map[task_id].op_idx).is_sink()) {
          // post_evaluate_driver()
          EvalWorkEntry post_evaluate_input_entry;
          TaskStream task_stream = task_stream_map[task_id];

          // fill in EvalWorkEntry manually
          EvalWorkEntry evaluate_input_entry;
          post_evaluate_input_entry.table_id = table_id;
          post_evaluate_input_entry.job_index = job_id;
          post_evaluate_input_entry.task_index = task_id;
          post_evaluate_input_entry.needs_configure = false; // hardcode by swjz
          post_evaluate_input_entry.needs_reset = false; // hardcode by swjz
          post_evaluate_input_entry.last_in_io_packet = true;
          post_evaluate_input_entry.last_in_task = true;
          // Op -> Elements
          std::map<i64, Elements> op_to_elements;
          // Op -> Device Handle
          std::map<i64, DeviceHandle> op_to_handle;
          // Op -> Rows
          std::map<i64, std::vector<i64>> op_to_row_ids;
          auto grab_source_start = now();
          for (i64 source_task : task_stream.source_tasks) {
            // The file name of eval_map[source_task]
            std::string eval_map_file_name = eval_map_path(bulk_job_id, source_task);
            std::unique_ptr<RandomReadFile> eval_map_input;
            BACKOFF_FAIL(
                make_unique_random_read_file(storage_, eval_map_file_name, eval_map_input),
                "while trying to make read file for " + eval_map_file_name);
            u64 pos = 0;
            auto read_start = now();
            std::vector<u8> eval_map_bytes = storehouse::read_entire_file(eval_map_input.get(), pos);
            scheduler_profiler.add_interval("read", read_start, now());
            proto::EvalWorkEntry post_input_proto;
            post_input_proto.ParseFromArray(eval_map_bytes.data(), eval_map_bytes.size());
            EvalWorkEntry eval_map_at_source_task;

            const TaskStream& source_task_stream = task_stream_map.at(source_task);
            DeviceHandle source_handle;
            if (ops.at(source_task_stream.op_idx).device_type() == proto::CPU) {
              source_handle = CPU_DEVICE;
            } else if (ops.at(source_task_stream.op_idx).device_type() == proto::GPU) {
              source_handle = {DeviceType::GPU, 0};
            }
            proto_to_evalworkentry(post_input_proto, eval_map_at_source_task,
                                   scheduler_profiler, device_handle);
            if (source_task_stream.op_idx == 0) {
              // The source task is a source Op. It might have multiple input columns
              // because we remapped input columns. So we skip merging process below.
              auto eval_work_entry = std::move(eval_map_at_source_task);
              post_evaluate_input_entry.columns.insert(post_evaluate_input_entry.columns.end(),
                                                       eval_work_entry.columns.begin(),
                                                       eval_work_entry.columns.end());
              post_evaluate_input_entry.column_handles.insert(post_evaluate_input_entry.column_handles.end(),
                                                              eval_work_entry.column_handles.begin(),
                                                              eval_work_entry.column_handles.end());
              post_evaluate_input_entry.row_ids.insert(post_evaluate_input_entry.row_ids.end(),
                                                       eval_work_entry.row_ids.begin(),
                                                       eval_work_entry.row_ids.end());
            } else {
              // We merge multiple columns from different tasks to one column
              // if these tasks come from the same op
              // Assume this source task only has one output column, it must be the last one
              size_t col_id = eval_map_at_source_task.columns.size() - 1;
              auto& column = eval_map_at_source_task.columns[col_id];
              auto& handle = eval_map_at_source_task.column_handles.at(col_id);
              // All source task data have been transferred to device_handle
              op_to_handle[source_task_stream.op_idx] = device_handle;
              op_to_row_ids[source_task_stream.op_idx].insert(
                  op_to_row_ids[source_task_stream.op_idx].end(),
                  task_stream.source_task_to_rows.at(source_task).begin(),
                  task_stream.source_task_to_rows.at(source_task).end());

              for (auto& row_id : task_stream.source_task_to_rows.at(source_task)) {
                for (size_t r = 0; r < eval_map_at_source_task.row_ids[col_id].size(); ++r) {
                  // Only need the columns that cover required row ids
                  if (row_id == eval_map_at_source_task.row_ids[col_id][r]) {
//                    add_element_ref(handle, column[r]);
                    op_to_elements[source_task_stream.op_idx].push_back(column[r]);
                  }
                }
              }
            }

//            // Since it has been used, we delete it from memory
//            for (auto i = 0; i < eval_map_at_source_task.columns.size(); ++i) {
//              auto& column = eval_map_at_source_task.columns.at(i);
//              auto& handle = eval_map_at_source_task.column_handles.at(i);
//              for (auto& element : column) {
//                delete_element(handle, element);
//              }
//            }
          }

          for (auto& kv : op_to_elements) {
            post_evaluate_input_entry.columns.push_back(kv.second);
            post_evaluate_input_entry.column_handles.push_back(op_to_handle.at(kv.first));
            post_evaluate_input_entry.row_ids.push_back(op_to_row_ids.at(kv.first));
          }

          scheduler_profiler.add_interval("grab_source", grab_source_start, now());

          // Only keep the last columns because we dropped liveness_analysis
          i32 num_sink_col = final_output_columns.size();
          VLOG(1) << "Output columns: " << num_sink_col;

          assert(num_sink_col <= post_evaluate_input_entry.columns.size());

          // Since it is not used, we delete it from memory
          for (auto column = post_evaluate_input_entry.columns.begin();
               column < post_evaluate_input_entry.columns.begin() +
               post_evaluate_input_entry.columns.size() - num_sink_col; ++column) {
            for (auto& element : *column) {
              delete_element(CPU_DEVICE, element);
            }
          }

          post_evaluate_input_entry.columns.erase(
              post_evaluate_input_entry.columns.begin(),
              post_evaluate_input_entry.columns.begin() +
              post_evaluate_input_entry.columns.size() - num_sink_col);

          post_evaluate_input_entry.column_handles.erase(
              post_evaluate_input_entry.column_handles.begin(),
              post_evaluate_input_entry.column_handles.begin() +
              post_evaluate_input_entry.column_handles.size() - num_sink_col);

          post_evaluate_input_entry.row_ids.erase(
              post_evaluate_input_entry.row_ids.begin(),
              post_evaluate_input_entry.row_ids.begin() +
              post_evaluate_input_entry.row_ids.size() -  num_sink_col);

          auto post_start = now();
          PostEvaluateWorkerArgs post_evaluate_worker_args {
              // Uniform arguments
              node_id_,

              // Per worker arguments
              0, eval_profilers[0][2], column_mapping.back(),
              final_output_columns, final_compression_options,
          };
          PostEvaluateWorker post_evaluate_worker(post_evaluate_worker_args);

          post_evaluate_worker.feed(post_evaluate_input_entry);
          EvalWorkEntry post_evaluate_output_entry;
          post_evaluate_worker.yield(post_evaluate_output_entry);

          scheduler_profiler.add_interval("post", post_start, now());

          EvalWorkEntry save_driver_input_entry = post_evaluate_output_entry;

          auto save_start = now();
          // save_driver()
          proto::Result save_worker_result;
          SaveWorkerArgs save_worker_args {
              node_id_,
              std::ref(sink_args),

              // Per worker arguments
              0, db_params_.storage_config,
              sink_factories, sink_configs,
              std::ref(save_thread_profilers[0]),
              std::ref(save_worker_result)
          };


          SaveWorker save_worker(save_worker_args);
          // NOTE(swjz): Force the task_id of SaveWorker to be 0 for now.
          save_worker.new_task(save_driver_input_entry.job_index, 0,
                               save_driver_input_entry.table_id, save_driver_input_entry.column_types);
          save_worker.feed(save_driver_input_entry);
          save_worker.finished();
          scheduler_profiler.add_interval("save", save_start, now());
        }
        // Regular (non-source/sink) Op
        else {
          const TaskStream& task_stream = task_stream_map[task_id];
          i32 op_idx = task_stream.op_idx;

          // fill in EvalWorkEntry manually
          EvalWorkEntry evaluate_input_entry;
          evaluate_input_entry.table_id = table_id;
          evaluate_input_entry.job_index = job_id;
          evaluate_input_entry.task_index = task_id;
          evaluate_input_entry.needs_configure = false; // hardcode by swjz
          evaluate_input_entry.needs_reset = false; // hardcode by swjz
          evaluate_input_entry.last_in_io_packet = true;
          evaluate_input_entry.last_in_task = true;
          // Op -> Elements
          std::map<i64, Elements> op_to_elements;
          // Op -> Device Handle
          std::map<i64, DeviceHandle> op_to_handle;
          // Op -> Rows
          std::map<i64, std::vector<i64>> op_to_row_ids;

          auto grab_source_start = now();
          for (i64 source_task : task_stream.source_tasks) {
            // The file name of eval_map[source_task]
            std::string eval_map_file_name = eval_map_path(bulk_job_id, source_task);
            std::unique_ptr<RandomReadFile> eval_map_input;
            BACKOFF_FAIL(
                make_unique_random_read_file(storage_, eval_map_file_name, eval_map_input),
                "while trying to make read file for " + eval_map_file_name);
            u64 pos = 0;
            std::vector<u8> eval_map_bytes = storehouse::read_entire_file(eval_map_input.get(), pos);
            proto::EvalWorkEntry evaluate_input_proto;
            evaluate_input_proto.ParseFromArray(eval_map_bytes.data(), eval_map_bytes.size());
            EvalWorkEntry eval_map_at_source_task;

            const TaskStream& source_task_stream = task_stream_map.at(source_task);
            DeviceHandle source_handle;
            if (ops.at(source_task_stream.op_idx).device_type() == proto::CPU) {
              source_handle = CPU_DEVICE;
            } else if (ops.at(source_task_stream.op_idx).device_type() == proto::GPU) {
              source_handle = {DeviceType::GPU, 0};
            }
            proto_to_evalworkentry(evaluate_input_proto, eval_map_at_source_task,
                                   scheduler_profiler, device_handle);
            if (source_task_stream.op_idx == 0) {
              // The source task is a source Op. It might have multiple input columns
              // because we remapped input columns. So we skip merging process below.
              auto eval_work_entry = std::move(eval_map_at_source_task);
              evaluate_input_entry.columns.insert(evaluate_input_entry.columns.end(),
                                                  eval_work_entry.columns.begin(),
                                                  eval_work_entry.columns.end());
              evaluate_input_entry.column_handles.insert(evaluate_input_entry.column_handles.end(),
                                                         eval_work_entry.column_handles.begin(),
                                                         eval_work_entry.column_handles.end());
              evaluate_input_entry.row_ids.insert(evaluate_input_entry.row_ids.end(),
                                                  eval_work_entry.row_ids.begin(),
                                                  eval_work_entry.row_ids.end());
            } else {
              // We merge multiple columns from different tasks to one column
              // if these tasks come from the same op
              // FIXME(swjz): Take the last one if source task has more than one output columns
              // This happens because we did not delete unused columns in evaluate_worker.cpp
//              assert(eval_map_at_source_task.columns.size() == 1);
              size_t col_id = eval_map_at_source_task.columns.size() - 1;
              auto& column = eval_map_at_source_task.columns[col_id];
              auto& handle = eval_map_at_source_task.column_handles.at(col_id);
              // All source task data have been transferred to device_handle
              op_to_handle[source_task_stream.op_idx] = device_handle;
              op_to_row_ids[source_task_stream.op_idx].insert(
                  op_to_row_ids[source_task_stream.op_idx].end(),
                  task_stream.source_task_to_rows.at(source_task).begin(),
                  task_stream.source_task_to_rows.at(source_task).end());

              for (auto& row_id : task_stream.source_task_to_rows.at(source_task)) {
                for (size_t r = 0; r < eval_map_at_source_task.row_ids[col_id].size(); ++r) {
                  // Only need the columns that cover required row ids
                  if (row_id == eval_map_at_source_task.row_ids[col_id][r]) {
//                    add_element_ref(handle, column[r]);
                    op_to_elements[source_task_stream.op_idx].push_back(column[r]);
                  }
                }
              }
            }

//            // Since it has been used, we delete it from memory
//            for (auto i = 0; i < eval_map_at_source_task.columns.size(); ++i) {
//              auto& column = eval_map_at_source_task.columns.at(i);
//              auto& handle = eval_map_at_source_task.column_handles.at(i);
//              for (auto& element : column) {
//                delete_element(handle, element);
//              }
//            }
          }

          for (auto& kv : op_to_elements) {
            evaluate_input_entry.columns.push_back(kv.second);
            evaluate_input_entry.column_handles.push_back(op_to_handle.at(kv.first));
            evaluate_input_entry.row_ids.push_back(op_to_row_ids.at(kv.first));
          }
          scheduler_profiler.add_interval("grab_source", grab_source_start, now());
          // Now evaluate_input_entry is ready to go!

          proto::Result evaluate_worker_result;
          auto eval_start = now();
          EvaluateWorkerArgs evaluate_worker_args {
              // Uniform arguments
              node_id_, startup_lock, startup_cv, startup_count,

              // Per worker arguments
              // NOTE(swjz): there is only one group if we use CPU only
              0, op_idx, groups[op_idx], job_params->boundary_condition(),
              eval_profilers[0][1], evaluate_worker_result
          };
          EvaluateWorker evaluate_worker(evaluate_worker_args);
          std::vector<TaskStream> task_stream_vector;
          task_stream_vector.push_back(task_stream);
          evaluate_worker.new_task(job_id, task_id, task_stream_vector);

          // the evaluate_input_entry here should contain all dependency data
          evaluate_worker.feed(evaluate_input_entry);
          EvalWorkEntry evaluate_output_entry;
          evaluate_worker.yield(work_packet_size, evaluate_output_entry);

          scheduler_profiler.add_interval("eval", eval_start, now());

          std::string eval_map_file_name = eval_map_path(bulk_job_id, task_id);
          std::unique_ptr<WriteFile> eval_map_output;
          BACKOFF_FAIL(
              make_unique_write_file(storage_, eval_map_file_name, eval_map_output),
              "while trying to make write file for " + eval_map_file_name);

          proto::EvalWorkEntry evaluate_output_proto;
          auto serialize_start = now();
          evalworkentry_to_proto(evaluate_output_entry, evaluate_output_proto,
              scheduler_profiler);
          scheduler_profiler.add_interval("serialize", serialize_start, now());
          size_t serialized_size = evaluate_output_proto.ByteSizeLong();

          std::vector<u8> eval_map_bytes;
          auto allocate_start = now();
          eval_map_bytes.resize(serialized_size);
          scheduler_profiler.add_interval("alloc", allocate_start, now());
          serialize_start = now();
          evaluate_output_proto.SerializeToArray(eval_map_bytes.data(), serialized_size);
          scheduler_profiler.add_interval("serialize", serialize_start, now());

          auto write_start = now();
          s_write(eval_map_output.get(), eval_map_bytes.data(), serialized_size);
          scheduler_profiler.add_interval("write", write_start, now());

          // Since it has been serialized, we delete it from memory
          for (auto i = 0; i < evaluate_output_entry.columns.size(); ++i) {
            auto& column = evaluate_output_entry.columns.at(i);
            auto& handle = evaluate_output_entry.column_handles.at(i);
            for (auto &element : column) {
              delete_element(handle, element);
            }
          }
        }

        allocated_work_to_queues[target_work_queue]++;
        accepted_tasks++;
        retired_tasks.enqueue(std::make_tuple(0, job_id, task_id));
      }
    }

    // Check for errors in pre-evaluate worker
    for (size_t i = 0; i < pre_eval_results.size(); ++i) {
      auto& result = pre_eval_results[i];
      if (!result.success()) {
        LOG(WARNING) << "(N/KI: " << node_id_ << "/" << i << ") "
                     << "pre-evaluate returned error result: " << result.msg();
        job_result->set_success(false);
        job_result->set_msg(result.msg());
        goto leave_loop;
      }
    }
    // Check for errors in evaluate worker
    for (size_t i = 0; i < eval_results.size(); ++i) {
      for (size_t j = 0; j < eval_results[i].size(); ++j) {
        auto& result = eval_results[i][j];
        if (!result.success()) {
          LOG(WARNING) << "(N/KI/KG: " << node_id_ << "/" << i << "/" << j
                       << ") returned error result: " << result.msg();
          job_result->set_success(false);
          job_result->set_msg(result.msg());
          goto leave_loop;
        }
      }
    }
    goto remain_loop;
  leave_loop:
    break;
  remain_loop:
    auto yield = now();
    std::this_thread::yield();
    scheduler_profiler.add_interval("yield", yield, now());
  }

  // If the job failed, can't expect queues to have drained, so
  // attempt to flush all queues here (otherwise we could block
  // on pushing into a queue)
  if (!job_result->success()) {
    load_work.clear();
    for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
      initial_eval_work[pu].clear();
    }
    for (i32 kg = 0; kg < num_kernel_groups; ++kg) {
      for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
        eval_work[pu][kg].clear();
      }
    }
    for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
      eval_work[pu].back().clear();
    }
    output_eval_work.clear();
    for (i32 i = 0; i < num_save_workers; ++i) {
      save_work[i].clear();
    }
    retired_tasks.clear();
  }

  auto push_exit_message = [](EvalQueue& q) {
    EvalWorkEntry entry;
    entry.job_index = -1;
    q.push(std::make_tuple(std::deque<TaskStream>(), entry));
  };

  auto push_output_eval_exit_message = [](OutputEvalQueue& q) {
    EvalWorkEntry entry;
    entry.job_index = -1;
    q.push(std::make_tuple(0, entry));
  };

  auto push_save_exit_message = [](SaveInputQueue& q) {
    EvalWorkEntry entry;
    entry.job_index = -1;
    q.push(std::make_tuple(0, entry));
  };

  // Push sentinel work entries into queue to terminate load threads
  for (i32 i = 0; i < num_load_workers; ++i) {
    LoadWorkEntry entry;
    entry.set_job_index(-1);
    load_work.push(
        std::make_tuple(0, std::deque<TaskStream>(), entry));
  }

  for (i32 i = 0; i < num_load_workers; ++i) {
    // Wait until all load threads have finished
//    load_threads[i].join();
  }

  // Push sentinel work entries into queue to terminate eval threads
  for (i32 i = 0; i < pipeline_instances_per_node; ++i) {
    if (distribute_work_dynamically) {
      push_exit_message(initial_eval_work[i]);
    } else {
      push_exit_message(initial_eval_work[0]);
    }
  }

  for (i32 i = 0; i < pipeline_instances_per_node; ++i) {
    // Wait until pre eval has finished
    LOG(INFO) << "Pre join " << i;
//    pre_eval_threads[i].join();
  }

  LOG(INFO) << "Exiting kernel threads";
  for (i32 kg = 0; kg < num_kernel_groups; ++kg) {
    for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
      push_exit_message(eval_work[pu][kg]);
    }
    for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
      // Wait until eval has finished
//      eval_threads[pu][kg].join();
    }
  }

  LOG(INFO) << "Exiting post-eval threads";
  // Terminate post eval threads
  for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
    push_exit_message(eval_work[pu].back());
  }
  for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
    // Wait until eval has finished
//    post_eval_threads[pu].join();
  }

  LOG(INFO) << "Exiting save threads";
  // Push sentinel work entries into queue to terminate coordinator thread
  push_output_eval_exit_message(output_eval_work);
//  save_coordinator_thread.join();

  // Push sentinel work entries into queue to terminate save threads
  for (i32 i = 0; i < num_save_workers; ++i) {
    // Wait until save thread is polling on save_work
    while(save_work[i].size() > 0) {
      retired_tasks.clear();
    }
    push_save_exit_message(save_work[i]);
  }
  for (i32 i = 0; i < num_save_workers; ++i) {
//    save_threads[i].join();
  }
  LOG(INFO) << "All threads are finished";

  // Ensure all files are flushed
  if (job_params->profiling()) {
    std::fflush(NULL);
    sync();
  }

  if (!job_result->success()) {
    finished_fn();
    return false;
  }

  // Write out total time interval
  timepoint_t end_time = now();

  // Check if we have any more allocations
  u64 max_mem_used = max_memory_allocated(CPU_DEVICE);
  u64 current_mem_used = current_memory_allocated(CPU_DEVICE);
  const auto& allocations = allocator_allocations(CPU_DEVICE);
  LOG(INFO) << "Max memory allocated:     " << max_mem_used / (1024 * 1024)
          << " MBs";
  LOG(INFO) << "Current memory allocated: " << current_mem_used / (1024 * 1024)
          << " MBs";
  if (num_load_workers > 0) {
    load_thread_profilers[0].increment("max_memory_used", max_mem_used);
    load_thread_profilers[0].increment("current_memory_used", current_mem_used);
  }
  VLOG(1) << "Leaked allocations: ";
  for (const auto& alloc : allocations) {
    VLOG(1) << alloc.call_file << ":" << alloc.call_line << ": refs "
            << alloc.refs << ", size " << alloc.size;
  }

  // Execution done, write out profiler intervals for each worker
  // TODO: job_name -> job_id?
  i32 job_id = meta.get_bulk_job_id(job_params->job_name());
  std::string profiler_file_name = bulk_job_worker_profiler_path(job_id, node_id_);
  std::unique_ptr<WriteFile> profiler_output;
  BACKOFF_FAIL(
      make_unique_write_file(storage_, profiler_file_name, profiler_output),
      "while trying to make write file for " + profiler_file_name);

  i64 base_time_ns =
      std::chrono::time_point_cast<std::chrono::nanoseconds>(base_time)
          .time_since_epoch()
          .count();
  i64 start_time_ns =
      std::chrono::time_point_cast<std::chrono::nanoseconds>(start_time)
          .time_since_epoch()
          .count();
  i64 end_time_ns =
      std::chrono::time_point_cast<std::chrono::nanoseconds>(end_time)
          .time_since_epoch()
          .count();
  s_write(profiler_output.get(), start_time_ns);
  s_write(profiler_output.get(), end_time_ns);

  i64 out_rank = node_id_;

  // Write process job profiler
  write_profiler_to_file(profiler_output.get(), out_rank, "process_job", "", 0,
                         profiler);

  // Load worker profilers
  u8 load_worker_count = num_load_workers;
  s_write(profiler_output.get(), load_worker_count);
  for (i32 i = 0; i < num_load_workers; ++i) {
    write_profiler_to_file(profiler_output.get(), out_rank, "load", "", i,
                           load_thread_profilers[i]);
  }

  // Evaluate worker profilers
  u8 eval_worker_count = pipeline_instances_per_node;
  s_write(profiler_output.get(), eval_worker_count);
  u8 profilers_per_chain = 3;
  s_write(profiler_output.get(), profilers_per_chain);
  for (i32 pu = 0; pu < pipeline_instances_per_node; ++pu) {
    i32 i = pu;
    {
      std::string tag = "pre";
      write_profiler_to_file(profiler_output.get(), out_rank, "eval", tag, i,
                             eval_profilers[pu][0]);
    }
    {
      std::string tag = "eval";
      write_profiler_to_file(profiler_output.get(), out_rank, "eval", tag, i,
                             eval_profilers[pu][1]);
    }
    {
      std::string tag = "post";
      write_profiler_to_file(profiler_output.get(), out_rank, "eval", tag, i,
                             eval_profilers[pu][2]);
    }
  }

  // Save worker profilers
  u8 save_worker_count = num_save_workers;
  s_write(profiler_output.get(), save_worker_count);
  for (i32 i = 0; i < num_save_workers; ++i) {
    write_profiler_to_file(profiler_output.get(), out_rank, "save", "", i,
                           save_thread_profilers[i]);
  }
  u8 scheduler_count = 1;
  s_write(profiler_output.get(), scheduler_count);
  write_profiler_to_file(profiler_output.get(), out_rank, "scheduler", "", 0,
                         scheduler_profiler);

  BACKOFF_FAIL(profiler_output->save(),
               "while trying to save " + profiler_output->path());

  std::fflush(NULL);
  sync();

  finished_fn();

  LOG(INFO) << "Worker " << node_id_ << " finished job";

  return true;
}

}
}