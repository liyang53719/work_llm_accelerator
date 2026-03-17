#ifndef TVM_WORK_LLM_ACCELERATOR_INCLUDE_AC_CHANNEL_H
#define TVM_WORK_LLM_ACCELERATOR_INCLUDE_AC_CHANNEL_H

template <class T>
class ac_channel {
public:
	typedef T element_type;
	enum { kCapacity = 8192 };

	ac_channel() : head_(0), tail_(0), count_(0) {}
	explicit ac_channel(int) : head_(0), tail_(0), count_(0) {}
	ac_channel(int, T) : head_(0), tail_(0), count_(0) {}

	T read() {
		T value = storage_[head_];
		head_ = (head_ + 1) % kCapacity;
		if (count_ > 0) {
			--count_;
		}
		return value;
	}

	void read(T& value) {
		value = read();
	}

	bool nb_read(T& value) {
		if (count_ == 0) {
			return false;
		}
		value = read();
		return true;
	}

	T peek() {
		return storage_[head_];
	}

	void peek(T& value) {
		value = peek();
	}

	bool nb_peek(T& value) {
		if (count_ == 0) {
			return false;
		}
		value = storage_[head_];
		return true;
	}

	void write(const T& value) {
		storage_[tail_] = value;
		tail_ = (tail_ + 1) % kCapacity;
		if (count_ < kCapacity) {
			++count_;
		}
	}

	bool nb_write(const T& value) {
		write(value);
		return true;
	}

	unsigned int size() const {
		return count_;
	}

	bool empty() const {
		return count_ == 0;
	}

	bool available(unsigned int count) const {
		return count_ >= count;
	}

	void reset() {
		head_ = 0;
		tail_ = 0;
		count_ = 0;
	}

private:
	T storage_[kCapacity];
	unsigned int head_;
	unsigned int tail_;
	unsigned int count_;
};

#endif