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

#pragma once

#include <grpc/support/log.h>
#include <grpc++/alarm.h>
#include "scanner/engine/rpc.grpc.pb.h"
#include "scanner/engine/runtime.h"
#include "scanner/engine/sampler.h"
#include "scanner/engine/dag_analysis.h"
#include "scanner/util/util.h"
#include "scanner/util/grpc.h"
#include "scanner/util/thread_pool.h"

#include <mutex>
#include <thread>

namespace scanner {
namespace internal {

using JobID = i64;
using WorkerID = i64;

template <class Request, class Reply>
using MCall = Call<MasterServerImpl, Request, Reply>;

class MasterServerImpl final : public proto::Master::Service {
 public:
  MasterServerImpl(DatabaseParameters& params, const std::string& port);

  ~MasterServerImpl();

  void run();

  void handle_rpcs();

  void start_watchdog(bool enable_timeout, i32 timeout_ms = 50000);

 private:
  void ShutdownHandler(MCall<proto::Empty, proto::Result>* call);

  // Database query methods
  void ListTablesHandler(MCall<proto::Empty, proto::ListTablesResult>* call);

  void GetTablesHandler(
      MCall<proto::GetTablesParams, proto::GetTablesResult>* call);

  void DeleteTablesHandler(
      MCall<proto::DeleteTablesParams, proto::Empty>* call);

  void NewTableHandler(
      MCall<proto::NewTableParams, proto::Empty>* call);

  void GetVideoMetadataHandler(MCall<proto::GetVideoMetadataParams,
                                     proto::GetVideoMetadataResult>* call);

  void IngestVideosHandler(
      MCall<proto::IngestParameters, proto::IngestResult>* call);

  // Worker methods
  void RegisterWorkerHandler(
      MCall<proto::WorkerParams, proto::Registration>* call);

  void UnregisterWorkerHandler(
      MCall<proto::UnregisterWorkerRequest, proto::Empty>* call);

  void ActiveWorkersHandler(
      MCall<proto::Empty, proto::RegisteredWorkers>* call);

  // Op and Kernel methods

  void GetOpInfoHandler(MCall<proto::OpInfoArgs, proto::OpInfo>* call);

  void GetSourceInfoHandler(
      MCall<proto::SourceInfoArgs, proto::SourceInfo>* call);

  void GetEnumeratorInfoHandler(
      MCall<proto::EnumeratorInfoArgs, proto::EnumeratorInfo>* call);

  void GetSinkInfoHandler(MCall<proto::SinkInfoArgs, proto::SinkInfo>* call);

  void LoadOpHandler(MCall<proto::OpPath, Result>* call);

  void RegisterOpHandler(MCall<proto::OpRegistration, proto::Result>* call);

  void RegisterPythonKernelHandler(
      MCall<proto::PythonKernelRegistration, proto::Result>* call);

  void ListLoadedOpsHandler(MCall<proto::Empty, proto::ListLoadedOpsReply>* call);

  void ListRegisteredOpsHandler(MCall<proto::Empty, proto::ListRegisteredOpsReply>* call);

  void ListRegisteredPythonKernelsHandler(
      MCall<proto::Empty, proto::ListRegisteredPythonKernelsReply>* call);

  void NextWorkHandler(
      MCall<proto::NextWorkRequest, proto::NextWorkReply>* call);

  void FinishedWorkHandler(
      MCall<proto::FinishedWorkRequest, proto::Empty>* call);

  void FinishedJobHandler(MCall<proto::FinishedJobRequest, proto::Empty>* call);

  void NewJobHandler(MCall<proto::BulkJobParameters, proto::NewJobReply>* call);

  void GetJobsHandler(MCall<proto::GetJobsRequest, proto::GetJobsReply>* call);

  void GetJobStatusHandler(
      MCall<proto::GetJobStatusRequest, proto::GetJobStatusReply>* call);

  // Misc methods
  void PingHandler(MCall<proto::Empty, proto::Empty>* call);

  void PokeWatchdogHandler(MCall<proto::Empty, proto::Empty>* call);

  // Expects context->peer() to return a string in the format
  // ipv4:<peer_address>:<random_port>
  // Returns the <peer_address> from the above format.
  static std::string get_worker_address_from_grpc_context(
      grpc::ServerContext* context);

  void recover_and_init_database();

  void start_job_processor();

  void stop_job_processor();

  bool process_job(const proto::BulkJobParameters* job_params,
                   proto::Result* job_result);

  void start_worker_pinger();

  void stop_worker_pinger();

  void start_job_on_workers(const std::vector<i32>& worker_ids);

  void stop_job_on_worker(i32 node_id);

  void remove_worker(i32 node_id);

  void blacklist_job(i64 job_id);

  DatabaseParameters db_params_;
  const std::string port_;

  std::unique_ptr<ThreadPool> pool_;
  std::thread pinger_thread_;
  std::atomic<bool> pinger_active_;
  std::condition_variable pinger_wake_cv_;
  std::mutex pinger_wake_mutex_;

  std::thread watchdog_thread_;
  std::atomic<bool> watchdog_awake_;
  Flag trigger_shutdown_;
  grpc::Alarm* shutdown_alarm_ = nullptr;
  storehouse::StorageBackend* storage_;
  DatabaseMetadata meta_;
  std::unique_ptr<TableMetaCache> table_metas_;
  std::vector<std::string> so_paths_;
  std::vector<proto::OpRegistration> op_registrations_;
  std::vector<proto::PythonKernelRegistration> py_kernel_registrations_;

  // Worker state
  WorkerID next_worker_id_ = 0;

  struct WorkerState {
    WorkerID id;
    bool active = false;
    std::unique_ptr<proto::Worker::Stub> stub;
    std::string address;
    // Tracks number of times the pinger has failed to reach a worker
    i64 failed_pings = 0;
  };

  std::map<WorkerID, std::shared_ptr<WorkerState>> workers_;

  // True if the master is executing a job
  std::mutex active_mutex_;
  std::condition_variable active_cv_;
  bool active_bulk_job_ = false;
  i32 active_bulk_job_id_ = 0;
  MCall<proto::BulkJobParameters, proto::NewJobReply>* new_job_call_;
  proto::BulkJobParameters job_params_;

  // True if all work for job is done
  std::mutex finished_mutex_;
  std::condition_variable finished_cv_;
  std::atomic<bool> finished_{true};

  std::thread job_processor_thread_;
  // Manages modification of all of the below structures
  std::mutex work_mutex_;

  // Worker connections
  std::map<std::string, i32> local_ids_;
  std::map<std::string, i32> local_totals_;

  enum struct BulkJobState {
    RUNNING,
    FINISHED
  };

  struct BulkJob {
    proto::BulkJobParameters job_params;
    BulkJobState state;

    //============================================================================
    // Preprocessed metadata about the supplied DAG
    //============================================================================
    DAGAnalysisInfo dag_info;
    // Mapping from jobs to table ids
    std::map<i64, i64> job_to_table_id;
    // Slice input rows for each job at each slice op
    std::vector<std::map<i64, i64>> slice_input_rows_per_job;
    // Number of output rows for each job
    std::vector<i64> total_output_rows_per_job;

    //============================================================================
    // Management of outstanding and completed jobs and tasks
    //============================================================================
    // The next job to use to generate tasks
    i64 next_job = 0;
    // Total number of jobs
    i64 num_jobs = -1;
    // Next task index in the current job
    i64 next_task = 0;
    // The number of bulk tasks in the current job
    i64 num_tasks = -1;
    // Job -> number of tasks
    std::vector<i64> tasks_per_job;

    // Outstanding set of generated task samples that should be processed
    // std::deque<job_id, task_id>
    std::deque<std::tuple<i64, i64>> unallocated_job_tasks;
    // The total number of tasks that have been completed
    std::atomic<i64> total_tasks_used{0};
    // The total number of tasks for this bulk job
    i64 total_tasks = 0;
    // The total number of tasks that have been completed for each job
    std::vector<i64> tasks_used_per_job;

    // Job -> Op -> maximum size for each task
    std::vector<std::map<i64, i64>> task_size_per_op;

    // Job -> Task -> TaskStream
    std::vector<std::map<i64, TaskStream>> task_streams;

    // Job -> Rows
    std::vector<std::vector<i64>> job_output_rows;

    Result task_result;

    //============================================================================
    // Assignment of tasks to workers
    //============================================================================
    // Tracks tasks assigned to worker so they can be reassigned if the worker
    // fails
    // Worker id -> (job_id, task_id)
    std::map<i64, std::set<std::tuple<i64, i64>>> active_job_tasks;
    // (Worker id, job_id, task_id) -> start_time
    std::map<std::tuple<i64, i64, i64>, double> active_job_tasks_starts;
    // Tracks number of times a task has been failed so that a job can be
    // removed if it is causing consistent failures job_id -> task_id ->
    // num_failures
    std::map<i64, std::map<i64, i64>> job_tasks_num_failures;
    // Tracks the jobs that have failed too many times and should be ignored
    std::set<i64> blacklisted_jobs;

    struct WorkerHistory {
      timepoint_t start_time;
      timepoint_t end_time;
      i64 tasks_assigned;
      i64 tasks_retired;
    };
    std::map<i64, WorkerHistory> worker_histories;
    std::map<i32, bool> unfinished_workers;
    std::vector<i32> unstarted_workers;
    std::atomic<i64> num_failed_workers{0};
    std::vector<i32> job_uncommitted_tables;
    
    Result job_result;
  };

  std::map<JobID, std::shared_ptr<BulkJob>> bulk_jobs_state_;

  std::unique_ptr<grpc::ServerCompletionQueue> cq_;
  proto::Master::AsyncService service_;
  std::unique_ptr<grpc::Server> server_;
};

}
}
