#include <iostream>
#include <unordered_map>
#include <list>
using namespace std;
class LRUCache {
public:
	/*
	* @param capacity: An integer
	*/LRUCache(int capacity) {
	// do intialization if necessary
	_capacity = capacity;
}

	  /*
	  * @param key: An integer
	  * @return: An integer
	  */
	  int get(int key) {
		  // write your code here
		  auto it = cache.find(key);
		  if (it != cache.end()) {
			  move_to_front(it);
			  return it->second.first;
		  }
		  else return -1;
	  }

	  /*
	  * @param key: An integer
	  * @param value: An integer
	  * @return: nothing
	  */
	  void set(int key, int value) {
		  // write your code here
		  auto it = cache.find(key);
		  if (it != cache.end()) {
			  move_to_front(it);
			  it->second.first = value;
		  }
		  else {
			  if (used.size() == _capacity) {
				  cache.erase(used.back());
				  used.pop_back();
			  }
			  used.push_front(key);
			  cache[key] = { value, used.begin() };

		  }
	  }
private:
	std::list<int> used;
	typedef pair<int, std::list<int>::iterator> PAIR;
	unordered_map<int, PAIR> cache;
	int _capacity;
	void move_to_front(unordered_map<int, PAIR>::iterator it) {
		int key = it->first;
		used.erase(it->second.second);
		used.push_front(key);
		it->second.second = used.begin();
	}
};