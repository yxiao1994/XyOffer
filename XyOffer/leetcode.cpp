#include <iostream>
#include <unordered_map>
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
	vector<int> twoSum(vector<int>& nums, int target) {
		vector<int> res;
		unordered_map<int, int> mapping;
		for (int i = 0; i < nums.size(); i++) {
			mapping[nums[i]] = i;
		}
		for (int i = 0; i < nums.size(); i++) {
			int x = target - nums[i];
			if (mapping.find(x) != mapping.end() && mapping[x] > i) {
				res.push_back(i);
				res.push_back(mapping[x]);
			}
		}
		return res;
	}
};