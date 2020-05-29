#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <unordered_map>
#include <algorithm>
using namespace std;
struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(NULL) {}
};
struct TreeNode {
	int val;
	TreeNode *left;
	TreeNode *right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution
{
	/**
	 * two sum
	 */
public:
	vector<int> twoSum(vector<int>& nums, int target) {
			vector<int> res;
			unordered_map<int, int> mapping;
			for (int i = 0; i < nums.size(); i++)
				mapping[nums[i]] = i;
			for(int i = 0; i < nums.size(); i++)
			{
				if(mapping.find(target - nums[i]) != mapping.end() && mapping[target - nums[i]] > i)
				{
					res.push_back(i);
					res.push_back(mapping[target - nums[i]]);
				}
			}
			return res;
		}
	/**
	 * 创建链表
	 */
public:
	ListNode* createList(vector<int>& nums)
	{
		if (nums.empty())
			return nullptr;
		ListNode* head = new ListNode(nums[0]);
		ListNode* curr = head;
		for(int i = 1; i < nums.size(); i++)
		{
			curr->next = new ListNode(nums[i]);
			curr = curr->next;
		}
		return head;
	}
	/**
	 * 输出链表
	 */
public:
	void printList(ListNode* head)
	{
		ListNode* curr = head;
		while(curr != nullptr)
		{
			cout << (curr->val) << '\t';
			curr = curr->next;
		}
		cout << endl;
	}
	/**
	 * 两个链表求和
	 */
public:
	ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
		ListNode* dummy = new ListNode(0);
		ListNode* curr = dummy;
		int sum = 0;
		while(l1 != NULL || l2 != NULL)
		{
			int val1 = (l1 != NULL) ? l1->val : 0;
			int val2 = (l2 != NULL) ? l2->val : 0;
			sum += (val1 + val2);
			curr->next = new ListNode(sum % 10);
			sum /= 10;
			if(l1 != NULL)
				l1 = l1->next;
			if(l2 != NULL)
				l2 = l2->next;
			curr = curr->next;
		}
		if (sum == 1)
			curr->next = new ListNode(1);
		return dummy->next;
	}
public:
	/**
	 * 不用加法做加法
	 */
	int aplusb(int a, int b) {
		// write your code here
		while (b != 0) {
			int sum = a ^ b;
			int carry = (a & b) << 1;
			a = sum;
			b = carry;
		}
		return a;
	}
public:
	/*
	 *
	 * @param : An integer
	 * @param : An integer
	 * @return: An integer denote the count of digit k in 1..n
	 */
	int digitCounts(int k, int n) {
		// write your code here
		int base = 1, res = 0;
		while (n >= base && k != 0) {
			int curr = (n / base) % 10;  //当前位
			int high = n / (base * 10);  // 高位
			int low = n % base;
			if (curr == k) {
				res += (high * base) + low + 1;
			}
			else if (curr < k)
				res += high * base;
			else res += (high + 1) * base;
			base *= 10;
		}
		if (k == 0) {
			res = 1;
			while (n >= base) {
				int high = n / (base * 10);
				res += high * base;
				base *= 10;
			}
		}
		return res;
	}
public:
	/*
	 *第k大的数
	 * @param n: An integer
	 * @param nums: An array
	 * @return: the Kth largest element
	 */
	int kthLargestElement(int k, vector<int> &nums) {
		// write your code here
		return kthSmallest(nums.size() + 1 - k, nums, 0, nums.size() - 1);
	}
private:
	/**
	 * 
	 */
	int kthSmallest(int k, vector<int> &nums, int p, int r)
	{
		int q = partition(nums, p, r);
		int qorder = q - p + 1;
		if (k == qorder)
			return nums[q];
		else if (k < qorder)
			return kthSmallest(k, nums, p, q - 1);
		else return kthSmallest(k - qorder, nums, q + 1, r);
	}
private:
	/**
	 * 
	 */
	int partition(vector<int> &nums, int p, int r)
	{
		int i = p - 1;
		for(int j = p; j < r; j++)
		{
			if(nums[j] < nums[r])
			{
				i++;
				swap(nums[i], nums[j]);
			}
		}
		swap(nums[i + 1], nums[r]);
		return i + 1;
	}
public:
	/**
	* 二叉树序列化
	*/
	string serialize(TreeNode * root) {
		// write your code here
		string s;
		if (root == NULL)
			return "#";
		return to_string(root->val) + "," + serialize(root->left) + "," + serialize(root->right);
	}

	/**
	* 二叉树反序列化
	*/
	TreeNode * deserialize(string &data) {
		// write your code here
		if (data == "#")
			return NULL;
		stringstream s(data);
		return helperDeserialize(s);
	}
private:
	TreeNode* helperDeserialize(stringstream& s)
	{
		string str;
		getline(s, str, ',');
		if (str == "#")
			return NULL;
		else
		{
			TreeNode* root = new TreeNode(stoi(str));
			root->left = helperDeserialize(s);
			root->right = helperDeserialize(s);
			return root;
		}
	}
public:
	/**
	 * 字符串旋转
	 * @param str: An array of char
	 * @param offset: An integer
	 */
	void rotateString(string &str, int offset) {
		// write your code here
		if (str.empty() || str.size() == 0)
			return;
		offset %= str.size();
		reverse(str, 0, str.size() - offset - 1);
		reverse(str, str.size() - offset, str.size() - 1);
		reverse(str, 0, str.size() - 1);
	}
private:
	void reverse(string& str, int start, int end)
	{
		for (int i = start, j = end; i < j; i++, j--)
			swap(str[i], str[j]);
	}
public:
	/*
	* @param source: source string to be scanned.
	* @param target: target string containing the sequence of characters to match
	* @return: a index to the first occurrence of target in source, or -1  if target is not part of source.
	*/
	int strStr(const char *source, const char *target) {
		// write your code here
		if (source == NULL || target == NULL)
			return -1;
		int len1 = strlen(source), len2 = strlen(target);
		if (len2 == 0)
			return 0;
		for (int i = 0; i <= len1 - len2; i++)
		{
			if (source[i] == target[0])
			{
				int j = 1;
				for (; j < len2; j++)
				{
					if (source[i + j] != target[j])
						break;
				}
				if (j == len2)
					return i;
			}
		}
		return -1;
	}
public:
	/*
	* @param nums: A list of integers.
	* @return: A list of permutations.
	*/
	vector<vector<int>> permute(vector<int> &nums) {
		// write your code here
		vector<vector<int>> res;
		vector<int> curr;
		vector<bool> visited(nums.size(), false);
		permutedfs(res, nums, curr, visited);
		return res;
	}
private:
	void permutedfs(vector<vector<int>> & res, vector<int> & nums, vector<int> & curr, vector<bool> & visited)
	{
		if (curr.size() == nums.size()) {
			res.push_back(curr);
			return;
		}
		for(int i = 0; i < nums.size(); i++)
		{
			if(!visited[i])
			{
				visited[i] = true;
				curr.push_back(nums[i]);
				permutedfs(res, nums, curr, visited);
				visited[i] = false;
				curr.pop_back();
			}
		}
	}
public:
	/*
	* @param nums: A list of integers.
	* @return: A list of permutations.
	*/
	vector<vector<int>> permuteII(vector<int> &nums) {
		// write your code here
		vector<vector<int>> res;
		permuteRecursive(nums, 0, res);
		return res;
	}
private:
	void permuteRecursive(vector<int> &nums, int begin, vector<vector<int>> & res) {
		if (begin == nums.size()) {
			res.push_back(nums);
			return;
		}
		for (int i = begin; i < nums.size(); i++) {
			swap(nums[i], nums[begin]);
			permuteRecursive(nums, begin + 1, res);
			swap(nums[i], nums[begin]);
		}
	}
public:
	/**
	* @param nums: A set of numbers
	* @return: A list of lists
	*/
	vector<vector<int>> subsets(vector<int> &nums) {
		// write your code here
		vector<vector<int>> res;
		vector<int> curr;
		res.push_back(curr);
		sort(nums.begin(), nums.end());
		dfs(res, curr, nums, 0);
		return res;
	}
private:
	void dfs(vector<vector<int>> & res, vector<int> & curr, vector<int> &nums, int start) {
		for (int i = start; i < nums.size(); i++) {
			curr.push_back(nums[i]);
			res.push_back(curr);
			dfs(res, curr, nums, i + 1);
			curr.pop_back();
		}
	}
public:
	/**
	* @param head: ListNode head is the head of the linked list
	* @param m: An integer
	* @param n: An integer
	* @return: The head of the reversed ListNode
	*/
	ListNode * reverseBetween(ListNode * head, int m, int n) {
		// write your code here
		ListNode* dummy = new ListNode(0);
		dummy->next = head;
		ListNode * curr = dummy;
		for(int i = 0; i < m - 1; i++)
		{
			curr = curr->next;
		}
		ListNode * tempHead = curr;
		curr = curr->next;
		ListNode * tempEnd = curr;
		tempHead->next = NULL;
		for(int i = 0; i <= n - m ;i++)
		{
			ListNode * pnext = curr->next;
			curr->next = tempHead->next;
			tempHead->next = curr;
			curr = pnext;
		}
		tempEnd->next = curr;
		return dummy->next;
	}
public:
	/**
	* @param nums: A list of integers
	* @return: An integer indicate the value of maximum difference between two substrings
	*/
	int maxDiffSubArrays(vector<int> &nums) {
		// write your code here
		int n = nums.size();
		int res = 0;
		vector<int> leftmax(n);
		vector<int> leftmin(n);
		leftmax[0] = leftmin[0] = nums[0];
		int localmax = nums[0], localmin = nums[0];
		for(int i= 1; i < n; i++)
		{
			localmax = max(localmax + nums[i], nums[i]);
			localmin = min(localmin + nums[i], nums[i]);
			leftmax[i] = max(leftmax[i - 1], localmax);
			leftmin[i] = min(leftmin[i - 1], localmin);
		}
		vector<int> rightmax(n);
		vector<int> rightmin(n);
		localmax = localmin = rightmax[n - 1] = rightmin[n-1] = nums[n - 1];
		for(int i = n - 2; i >= 0; i--)
		{
			localmax = max(nums[i], localmax + nums[i]);
			localmin = min(nums[i], localmin + nums[i]);
			rightmax[i] = max(rightmax[i + 1], localmax);
			rightmin[i] = min(rightmin[i + 1], localmin);
		}
		for(int i = 0; i < n-1; i++)
		{
			res = max(res, max(abs(leftmax[i] - rightmin[i + 1]), abs(rightmax[i + 1] - leftmin[i])));
		}
		return res;
	}
public:
	/*
	* @param s: A string
	* @return: A list of lists of string
	*/
	vector<vector<string>> partition(string &s) {
		// write your code here
		vector<vector<string>> res;
		if (s.size() == 0)
			return res;
		vector<string> curr;
		partitionDFS(res, curr, s, 0);
		return res;
	}
private:
	void partitionDFS(vector<vector<string>> & res, vector<string> & curr, string & s, int start)
	{
		if(start == s.size())
		{
			res.push_back(curr);
			return;
		}
		for(int i = start; i < s.size(); i++)
		{
			string str = s.substr(start, i - start + 1);
			if(ispalindrome(str))
			{
				curr.push_back(str);
				partitionDFS(res, curr, s, i + 1);
				curr.pop_back();
			}
		}
	}
private:
	bool ispalindrome(string & s)
	{
		int i = 0, j = s.size() - 1;
		while(i < j)
		{
			if (s[i] != s[j])
				return false;
			i++;
			j--;
		}
		return true;
	}
public:
	/**
	* @param nums: An integer array
	* @return: The length of LIS (longest increasing subsequence)
	*/
	int longestIncreasingSubsequence(vector<int> &nums) {
		// write your code here
		vector<int> dp;
		for(int i = 0; i < nums.size(); i++)
		{
			auto it = lower_bound(dp.begin(), dp.end(), nums[i]);
			if (it != dp.end())
				*it = nums[i];
			else dp.push_back(nums[i]);
		}
		return dp.size();
	}
/**
 * 所有的递增子序列
 */
public:
	vector<vector<int>> allIS(vector<int> & nums)
	{
		return ISRecursive(nums, nums.size() - 1);
	}
private:
	vector<vector<int>> ISRecursive(vector<int> & nums, int index)
	{
		vector<vector<int>> res;
		if (index <= 0)
			return res;
		//index左边（不包含index）的所有递增子序列
		vector<vector<int>> left = ISRecursive(nums, index - 1);
		// 首先放入以nums[index]结尾且长度为2的子序列
		for(int i = 0; i < index; i++)
		{
			if (nums[index] > nums[i])
				res.push_back({nums[i], nums[index]});
		}
		// index-1左边的递增子序列
		for (auto vec : left) {
			res.push_back(vec);
		}
		// 以nums[index]结尾的递增子序列（长度大于2）
		for(auto vec : left)
		{
			if(nums[index] > vec.back())
			{
				vec.push_back(nums[index]);
				res.push_back(vec);
			}
		}
		return res;
	}
public:
	/**
	* @param str: String
	* @return: String
	*/
	string convertPalindrome(string &str) {
		// Write your code here
		int i = 0, j = str.size() - 1;
		for (; j >= 0; j--) {
			if (str[i] == str[j])
				i++;
		}
		if (i == str.size())
			return str;
		string suffix = str.substr(i);
		return string(suffix.rbegin(), suffix.rend()) + convertPalindrome(str.substr(0, i)) + suffix;
	}
public:
	int maximalRectangle(vector<vector<char>>& matrix) {
		if (matrix.size() == 0 || matrix[0].size() == 0)
			return 0;
		int m = matrix.size(), n = matrix[0].size();
		int res = 0;
		vector<vector<int>> dp(m, vector<int>(n, 0));
		for (int i = 0; i < m; i++)
			if (matrix[i][0] == '1')
				dp[i][0] = 1;
		for(int i = 0; i < m ;i++)
		{
			for(int j = 1; j < n; j++)
			{
				if (matrix[i][j] == '1')
					dp[i][j] = dp[i][j-1] + 1;
			}
		}
		for(int i = 0; i < m; i++)
		{
			for(int j = 0; j < n; j++)
			{
				int row1 = i - 1, row2 = i + 1;
				while (row1 >= 0 && dp[row1][j] >= dp[i][j])
					row1--;
				while (row2 < m && dp[row2][j] >= dp[i][j])
					row2++;
				res = max(res, (row2 - row1 - 1) * dp[i][j]);
			}
		}
		return res;
	}
public:
	/**
	* @param head: The head of linked list.
	* @return: You should return the head of the sorted linked list, using constant space complexity.
	*/
	ListNode * MergeSortList(ListNode * head) {
		// write your code here
		if (head == NULL || head->next == NULL)
			return head;
		ListNode* mid = midOfList(head);
		ListNode* right = mid->next;
		mid->next = NULL;
		MergeSortList(head);
		MergeSortList(right);
		return mergeTwoSortedList(head, right);
	}
private:
	ListNode * midOfList(ListNode * head) {
		ListNode * fast = head;
		ListNode * slow = head;
		while (fast->next != NULL && fast->next->next != NULL) {
			fast = fast->next->next;
			slow = slow->next;
		}
		return slow;
	}
	ListNode* mergeTwoSortedList(ListNode * head1, ListNode * head2) {
		ListNode * dummy = new ListNode(0);
		ListNode * curr = dummy;
		while (head1 != NULL && head2 != NULL) {
			if (head1->val < head2->val) {
				curr->next = head1;
				head1 = head1->next;
			}
			else {
				curr->next = head2;
				head2 = head2->next;
			}
			curr = curr->next;
		}
		if (head1 != NULL)
			curr->next = head1;
		if (head2 != NULL)
			curr->next = head2;
		return dummy->next;
	}
public:
	/**
	 * 输出最长公共子序列
	 */
	void printLCS(string & s1, string & s2)
	{
		int m = s1.size(), n = s2.size();
		vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
		//用于标记方向的数组，0表示左上，1表示向左，2表示向上
		vector<vector<int>> flag(m + 1, vector<int>(n + 1));
		for(int i = 1; i <= m; i++){
			for(int j = 1; j <= n; j++)
			{
				if(s1[i-1] == s2[j-1])
				{
					dp[i][j] = dp[i - 1][j - 1] + 1;
					flag[i][j] = 0;
				}
				else if(dp[i-1][j] < dp[i][j-1])
				{
					dp[i][j] = dp[i][j-1];
					flag[i][j] = 1;
				}
				else
				{
					dp[i][j] = dp[i-1][j];
					flag[i][j] = 2;
				}
			}
		}
		LCS_PRINT(dp, flag, s1, s2, m, n);
	}
private:
	void LCS_PRINT(vector<vector<int>> &dp, vector<vector<int>> &flag, string & s1, string & s2, int row, int col)
	{
		if (row <= 0 || col <= 0)
			return;
		if (flag[row][col] == 0) {//左上
			LCS_PRINT(dp, flag, s1, s2, row - 1, col - 1);
			cout << s1[row - 1];
		}
		if(flag[row][col] == 1)
			LCS_PRINT(dp, flag, s1, s2, row, col - 1);
		if (flag[row][col] == 2)
			LCS_PRINT(dp, flag, s1, s2, row - 1, col);
	}
public:
	/**
	 * 矩阵中的最长递增路径
	 */
	int longestIncreasingPath(vector<vector<int>>& matrix) {
		if (matrix.size() == 0 || matrix[0].size() == 0)
			return 0;
		int m = matrix.size(), n = matrix[0].size();
		vector<vector<int>> dp(m, vector<int>(n, 0));
		int res = 0;
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				res = max(res, dfs(matrix, dp, i, j, m, n));
		return res + 1;
	}
private:
	int dfs(vector<vector<int>>& matrix, vector<vector<int>>& dp, int i, int j, int m, int n) {

		if (dp[i][j] != 0)
			return dp[i][j];
		vector<int> dx{ 0, 0, -1, 1 };
		vector<int> dy{ 1, -1, 0, 0 };
		int res = 0;
		for (int k = 0; k < 4; k++) {
			int x = i + dx[k];
			int y = j + dy[k];
			if (x < 0 || x >= m || y < 0 || y >= n || matrix[i][j] <= matrix[x][y])
				continue;
			res = max(res, dfs(matrix, dp, x, y, m, n) + 1);
		}
		dp[i][j] = res;
		return res;
	}
public:
	/**
	* @param head: The head of linked list.
	* @return: You should return the head of the sorted linked list, using constant space complexity.
	*/
	ListNode * QuicksortList(ListNode * head) {
		// write your code here
		if (head == NULL || head->next == NULL)
			return head;
		ListNode* pend = head;
		while (pend->next != NULL)
			pend = pend->next;
		QuickSort(head, pend);
		return head;
	}
private:
	void QuickSort(ListNode* head, ListNode* end) {
		if (head != end) {
			ListNode* p = partition(head, end);
			QuickSort(head, p);
			if (p->next != NULL)
				QuickSort(p->next, end);
		}
	}
	ListNode* partition(ListNode* head, ListNode* end) {
		int key = head->val;
		ListNode* p = head;
		ListNode* q = head->next;
		while (q != end->next) {
			if (q->val < key) {
				p = p->next;
				swap(p->val, q->val);
			}
			q = q->next;
		}
		swap(p->val, head->val);
		return p;
	}
};


enum color
{
	red, green, blue
};


int main() {
	Solution obj;
	vector<int> nums = {1, 3, 2, 4};
	vector<vector<int>> res = obj.allIS(nums);
	vector<vector<int>> matrix{ {9, 9, 4}, { 6, 6, 8 }, { 2, 1, 1 }};
	//cout << obj.longestIncreasingPath(matrix);
	/*
	for(vector<int> vec:res)
	{
		for (int a : vec)
			cout << a << " ";
		cout << endl;
	}*/
	string s1("abcdeefg");
	string s2("aceedfg");
	obj.printLCS(s1, s2);
	system("pause");
	return 0;
}