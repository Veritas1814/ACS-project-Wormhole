#include "thread_pool.h"

ThreadPool::ThreadPool(size_t num_threads) : stop(false) {
	for (size_t i = 0; i < num_threads; ++i) {
		threads.emplace_back([this]() {
			while (true) {
				std::function<void()> task;
				{
					std::unique_lock<std::mutex> lock(queue_mutex);
					condition.wait(lock, [this]() { return stop || !tasks.empty(); });

					if (stop && tasks.empty()) return;

					task = std::move(tasks.front());
					tasks.pop();
				}
				task();
			}
		});
	}
}

ThreadPool::~ThreadPool() {
	{
		std::unique_lock<std::mutex> lock(queue_mutex);
		stop = true;
	}
	condition.notify_all();
	for (std::thread &worker : threads) {
		worker.join();
	}
}
