#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>

class ThreadPool {
public:
	explicit ThreadPool(size_t num_threads);
	~ThreadPool();

	template<class F, class... Args>
	auto submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
		auto task = std::make_shared<std::packaged_task<decltype(f(args...))()>>(
			std::bind(std::forward<F>(f), std::forward<Args>(args)...)
		);

		std::future<decltype(f(args...))> res = task->get_future();

		{
			std::lock_guard<std::mutex> lock(queue_mutex);
			if (stop)
				throw std::runtime_error("ThreadPool is stopped");
			tasks.emplace([task]() { (*task)(); });
		}

		condition.notify_one();
		return res;
	}

private:
	std::vector<std::thread> threads;
	std::queue<std::function<void()>> tasks;

	std::mutex queue_mutex;
	std::condition_variable condition;
	bool stop;
};
