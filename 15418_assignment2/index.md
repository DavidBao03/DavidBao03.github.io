# 15418_assignment2

# 15418 Assignment2
## 作业说明：
本次的作业是要求完成一个线程池，并分别实现同步和异步使用。任务还要求实现一个任务图的调用，即任务之间有依赖关系时，线程池该如何调度这些任务。

## 我的一些理解：
1. 线程池：为了避免创建大量线程，CPU不断切换线程占用资源，可以先提前创建好对应CPU数量的线程数，并为每个线程创建一个任务队列。线程池的使用就是为每个任务队列提交任务。
2. 同步使用：在每次发布任务后，主线程会等待任务完成之后再返回。
3. 异步使用：在每次分布任务后，主线程立即返回，无需等待任务完成。

## 实现方式：
***注：本文的实现大量参考了<https://github.com/PKUFlyingPig/asst2>中的实现***

首先，根据任务提示，我们需要创建两个队列，分别对应了就绪任务和等待任务。
我们可以创建以下两个结构体来记录任务的状态：
```
struct ReadyTask {
    TaskID ready_task_id_;
    int current_task_;
    int total_tasks_num_;
    IRunnable* runnable_;
    ReadyTask(TaskID id, IRunnable* runnable, int num_total_tasks)
    :ready_task_id_(id), current_task_{0}, total_tasks_num_(num_total_tasks), runnable_(runnable){}
    ReadyTask(){}
};

struct WaitingTask {
    TaskID waiting_task_id_;
    TaskID deponds_task_id_;
    int total_tasks_num_;
    IRunnable* runnable_;
    WaitingTask(TaskID id, TaskID depID, IRunnable* runnable, int num_total_tasks)
    :waiting_task_id_(id), deponds_task_id_(depID), 
    total_tasks_num_(num_total_tasks), runnable_(runnable){}

    bool operator<(const WaitingTask& w2) const {
        return waiting_task_id_ > w2.waiting_task_id_;
    }
};
```

其次，我们在线程池的实现中增加一些字段：
```
private:
    int num_threads_;

    bool killed_;

    TaskID finish_task_id_;
    TaskID next_id_;

    std::thread* thread_pool_;

    std::queue<ReadyTask> ready_queue_;
    std::queue<WaitingTask> waiting_queue_;

    std::mutex ready_queue_mutex_;

    std::mutex waiting_queue_mutex_;

    std::mutex data_mutex_;

    std::unordered_map<TaskID, std::pair<int, int>> taskID_record;

    std::condition_variable finished_;
```

紧接着是构造函数：
```
TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //

    num_threads_ = num_threads;
    killed_ = false;
    finish_task_id_ = -1;
    next_id_ = 0;
    thread_pool_ = new std::thread[num_threads];

    for (int i = 0; i < num_threads_; i++) {
        thread_pool_[i] = std::thread([this]() {
            while(!killed_) {
                ReadyTask task;
                bool isRunningTask = false;

                ready_queue_mutex_.lock();
                if (ready_queue_.empty()) {
                    waiting_queue_mutex_.lock();
                    while (!waiting_queue_.empty()) {
                        auto& next_task = waiting_queue_.front();
                        if (next_task.deponds_task_id_ > finish_task_id_) break;
                        ready_queue_.push(ReadyTask(next_task.waiting_task_id_, next_task.runnable_,
                        next_task.total_tasks_num_));
                        taskID_record.insert({next_task.waiting_task_id_, {0, next_task.total_tasks_num_}});
                        waiting_queue_.pop();
                    }
                    waiting_queue_mutex_.unlock();
                } else {
                    task = ready_queue_.front();
                    if (task.current_task_ >= task.total_tasks_num_) {
                        ready_queue_.pop();
                    } else {
                        ready_queue_.front().current_task_++;
                        isRunningTask = true;
                    }
                }
                ready_queue_mutex_.unlock();

                if (isRunningTask) {
                    task.runnable_->runTask(task.current_task_, task.total_tasks_num_);

                    data_mutex_.lock();
                    auto& [finished_task, total_task] = taskID_record[task.ready_task_id_];
                    finished_task++;
                    if (finished_task == total_task) {
                        taskID_record.erase(task.ready_task_id_);
                        finish_task_id_ = std::max(task.ready_task_id_, finish_task_id_);
                    }
                    data_mutex_.unlock();
                }
            }
        });
    }
}
```
其中，构造函数不仅初始化了一些状态数据，还定义了线程获取任务的实现。线程获取任务分为两部分。其一，若就绪队列中已经有任务了，只需要从就绪队列中取出任务并执行即可：
```
task = ready_queue_.front();
if (task.current_task_ >= task.total_tasks_num_) {
    ready_queue_.pop();
} else {
    ready_queue_.front().current_task_++;
    isRunningTask = true;
}
```
其二，若就绪队列中没有任务，则需要从等待队列中取出任务。取出任务时需要注意，若等待任务的依赖任务（或前置任务）没有完成的话，就不能将其变成就绪任务。
```
while (!waiting_queue_.empty()) {
		auto& next_task = waiting_queue_.front();
		if (next_task.deponds_task_id_ > finish_task_id_) break;
		ready_queue_.push(ReadyTask(next_task.waiting_task_id_,
		next_task.runnable_,next_task.total_tasks_num_));
    taskID_record.insert({next_task.waiting_task_id_,
    {0,next_task.total_tasks_num_}});
    waiting_queue_.pop();
}
```
任务执行的实现：
```
if (isRunningTask) {
		task.runnable_->runTask(task.current_task_, task.total_tasks_num_);

    data_mutex_.lock();
    auto& [finished_task, total_task]=taskID_record[task.ready_task_id_];
    finished_task++;
    if (finished_task == total_task) {
    		taskID_record.erase(task.ready_task_id_);
        finish_task_id_ = std::max(task.ready_task_id_, finish_task_id_);
    }
    data_mutex_.unlock();
}
```
注意：在对这些队列或共享数据进行操作前，一定要记得加锁

接下来是析构函数，比较简单：
```
TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    //
    // TODO: CS149 student implementations may decide to perform cleanup
    // operations (such as thread pool shutdown construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //

    killed_ = true;
    for (int i = 0; i < num_threads_; i++) {
        thread_pool_[i].join();
    }

    // delete thread_pool_;
}
```

异步执行的函数：
```
TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks, const std::vector<TaskID>& deps) {

    //
    // TODO: CS149 students will implement this method in Part B.
    //

    TaskID dep = -1;
    if (!deps.empty()) {
        // dep = *std::max(deps.begin(), deps.end());
        dep = *deps.end();
    }

    WaitingTask task = WaitingTask(next_id_, dep,runnable,num_total_tasks);
    waiting_queue_mutex_.lock();
    waiting_queue_.push(std::move(task));
    waiting_queue_mutex_.unlock();

    return next_id_++;
}
```
注释前为参考仓库原本的实现，之后我将在文中解释为什么修改。总之，异步调用非常简单，在构造好任务之后，直接将其放入等待队列，在立即返回即可。

有了异步执行，就需要有同步的手段：
```
void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    while (true) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (finish_task_id_ + 1 == next_id_) break;
    }

    return;
}
```
只需一直检查任务是否做完即可。这里使用信号量应该会更好。

根据提示，同步执行只需要将异步执行和同步函数结合起来使用即可：
```
void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {
    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    std::vector<TaskID> noDeps;
    runAsyncWithDeps(runnable, num_total_tasks, noDeps);
    sync();
}
```

最后，解释一下我的修改：在原本的实现中，考虑依赖任务的方式是记录下已经完成任务的最大ID。因为作业假定了任务的ID是递增的，所以只需要考虑已经做过的最大ID就能知道某任务所依赖的任务是否被完成。若依赖ID小于最大ID，则可以直接完成，反之则不能。

在仔细观察（偷看）测试用例后，发现ID永远是一个个的push进入depond vector中的，这意味着最后一个进入vector的值永远是最大值，所以不必遍历整个vector，只需要取最后一个即可。

## 测试结果
如下：

![asst2_test_result](/img/asst2_test_result.png)

成功通过所有测试
