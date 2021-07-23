/**
 * date: 2018-6-24
 * author: zlm
 * Leetcode top 100 && nowcoder && jianzhioffer
 */
#include <iostream>
#include <cmath>
#include <string>
#include <cstring>
#include <queue>
#include <deque>
#include <stack>
#include <numeric>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <climits>
#include <utility>
#include <functional>
using namespace std;


struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

struct ListNode {
     int val;
     ListNode *next;
     ListNode(int x) : val(x), next(NULL) {}
};

// 1.Two Sum
class Solution {
public:
  vector<int> twoSum(vector<int> &numbers, int target) {
    if (numbers.empty())
      return vector<int>();
    vector<int> indice;
    
    map<int, int> hash;
    int len = numbers.size();
    for (int i = 0; i < len; i++) {
      hash[numbers[i]] = i;
    }
    for (int i = 0; i < len; i++) {
      int sub = target - numbers[i];
      // 题目要求满足条件的两个数必须不同，故不需要考虑同一个元素加上自身的情况
      if (hash.find(sub) != hash.end() && hash[sub] > i) {
        indice.push_back(i + 1);
        indice.push_back(hash[sub] + 1);
        break;
      }
    }
    return indice;
  }
};

// improved version
vector<int> twoSum2(vector<int> &num, int target) {
  if (num.empty())
    return vector<int>();
  vector<int> res(2);
  unordered_map<int, int> mp;
  int len = num.size();
  for (int i = 0; i < len; i++) {
    int sub = target - num[i];
    if (mp.find(sub) != mp.end()) { // 已经找到了两个元素
      res[0] = mp[sub];
      res[1] = mp[num[i]];
      break;
    }
    else
      mp.insert({num[i], i}); // mp[num[i]] = i;
  }
}

// 2.reverse list
// 使用循环实现(no head node)
class Solution7 {
public:
  ListNode* reverseList(ListNode* head) {
    if (head == nullptr || head->next == nullptr) // 只有一个结点或者为空
      return head;

    //ListNode *newHead = new ListNode(0);
    ListNode *pre = head;
    ListNode *pcur = head->next;
    ListNode *pnext = nullptr;

    while (pcur != nullptr) {
      pnext = pcur->next;
      pcur->next = pre; // 改变指向
      
      pre = pcur;
      pcur = pnext;
    } // end 此时pre指向最后一个结点，即反向后的第一个结点
    head->next = nullptr;
    return pre;
  }
};

// 3.带环链表的入口结点
// fast and slow pointer
class Solution8 {
public:
  ListNode *detectCycle(ListNode *head) {
    if (head == nullptr || head->next == nullptr)
      return nullptr;

    ListNode *fast = head;
    ListNode *slow = head;

    while (fast != nullptr && fast->next != nullptr) {
      fast = fast->next->next;
      slow = slow->next;
      if (slow == fast)
        break;
    }
     // 正常退出while循环，说明没有环存在
    if (fast == nullptr || fast->next == nullptr)
      return nullptr;
    // 否则：slow==fast，存在环，设置slow = head，当slow移动到入口结点，则fast正好移动至入口结点
    slow = head;
    while (slow != fast) {
      slow = slow->next;
      fast = fast->next;
    }
    return fast;
  }
};

// 5.数组中第k大的元素
// 思路：把数据从大到小排列，从前向后遍历寻找第k大的元素
// 不建议这种方法，可通过小顶堆来实现
class Solution9 {
public:
  int findKthLargest(vector<int>& nums, int k) {
    if (k < 1 || k > (int)nums.size() || nums.empty())
      return 0;

    sort(nums.begin(), nums.end());
    reverse(nums.begin(), nums.end()); // 把序列从大到小排列
    int len = nums.size();

    return nums[k - 1];
  }
};

/** 题目 347. Top K Frequent Elements
 * @author: zlm
 * @date: 2018-6-25 10：40
 * @description: 统计前k个最频繁出现的数字
 */
class Solution10 {
public:
  vector<int> topKFrequent(vector<int>& nums, int k) {
    vector<int> res;
    if (nums.empty())
      return res;
    
    unordered_map<int, int> mp;
    for (auto v : nums) {
      mp[v]++;
    }
    if (k > mp.size())
      return res;
    priority_queue<pair<int, int>> que; // 使用优先队列（基于红黑树实现，队头元素top最大）
    
    // 按元素出现频次进行排序
    for (auto it = mp.begin(); it != mp.end(); it++) {
      que.push({it->second, it->first });
    }

    // 队首存放的是次数最多的元素
    while (k--) {
      res.push_back(que.top().second);
      que.pop();
    }
    return res;
  }
};

/** 347. Top K Frequent Elements
 * @author: zlm
 * @date: 2018-6-25 10：40
 * @description: 桶排序思路，每个次数对应一个桶
 */
class Solution347 {
public:
  vector<int> topKFrequent(vector<int>& nums, int k) {
    map<int, int> m;
    vector<vector<int>> bucket(nums.size() + 1);
    vector<int> res;
    for (auto a : nums) 
      ++m[a];

    for (auto it : m) {
      bucket[it.second].push_back(it.first);
    }
    for (int i = nums.size(); i >= 0; --i) {
      for (int j = 0; j < bucket[i].size(); ++j) {
        res.push_back(bucket[i][j]);
        if (res.size() == k) 
          return res;
      }
    }
    return res;
  }
};

/** 题目 169. Majority Element
 * @author: zlm
 * @date: 2018-6-25 11：40
 * @description: 数组中出现次数超过一半的元素
 * @method: 对数组进行排序，超过一半的元素一定会出现在序列中间，即为数学上的中位数
 * 所以问题转换为求解第 N/2 大的元素。
 * 使用快速排序的 partition 思想，每次划分后，若基准元素的下标恰好为n/2，
 * 则该基准元素即为第n/2大元素。
 * 使用排序法在牛客上可以AC，但在leetcode中只能通过43/44个cases 
 */
class Solution12 {
public:
  int majorityElement(vector<int>& nums) {
    if (nums.empty())
      return -1;
    int N = nums.size();
    int low = 0, high = N - 1;
    
    while (low <= high) {
      int mid = partition(nums, low, high);
      if (mid == N / 2)  // mid 处于正中间位置，则一定是出现次数超过一半的那个元素
        return nums[mid];
      else if (mid < N / 2) {
        low = mid + 1;
      }
      else
        high = mid - 1;
    }
    // 这里还需要检查mid代表的元素出现次数是否超过一半
    return -1;
  }

  int partition(vector<int> &num, int low, int high) {
    // int low = 0, high = num.size() - 1;
    int pivot = num[low];
    while (low < high) {
      while (low < high && pivot <= num[high])
        high--;
      num[low] = num[high];

      while (low < high && pivot > num[low])
        low++;
      num[high] = num[low];
    }
    num[low] = pivot;
    return low;
  }
};

// 另一种解法：
// 1. 设置一个计数器，选择第一个元素为当前元素；
// 2. 如果下一个元素与当前元素相同，cnt++，否则cnt--；
// 3. 若cnt==0，则更改当前元素，并令cnt=1；
// 4. 最终cnt不为0，返回对应的当前元素
// 如果 majority 存在，则经过抵消和增加之后，cnt一定>=1;
class Solution13 {
public:
  int majorityElement(vector<int>& nums) {
    if (nums.empty())
      return -1;
  
    int cnt = 1;
    int curElem = nums[0]; // 选择第一个元素用于和其他元素相抵消
    for (int i = 1; i < nums.size(); i++) {
      if (cnt != 0) {
        if (curElem == nums[i]) 
          cnt++;
        else
          cnt--;
      }
      else { // cnt == 0 时，需要重新选择当前元素并计数
        curElem = nums[i];
        cnt = 1;
      }
    }
    // 还需要做一次验证，因为测试用例中可能不存在majority
    return curElem; // 最终cnt>=1，此时经过抵消后，curElem即为出现次数过半的元素
  }
};


/** 287. Find the Duplicate Number
 * @author: zlm
 * @date: 2018-6-25 11：40
 * @description: 找出数组中非重复的元素（给定数组只有一个元素是非重复的，其他均为重复元素）
 * 要求不能修改数组，只能读取数组元素，空间复杂度O(1）
 * @method: 异或
 */
class Solution287 {
public:
  int findDuplicate(vector<int>& nums) {
    if (nums.size() == 0)
      return 0;
    int res = 0;
    for (int i = 0; i < (int)nums.size(); i++) {
      res ^= nums[i];
    }
    return res;
  }
};

/** 5. Longest Palindromic Substring 
 * @author: zlm
 * @date: 2018-6-25 22：12
 * @description: 找出字符串中最长的回文子串
 * @method: 动态规划法, 时间和空间复杂度均为 n^2
 */
class Solution15 {
public:
  string longestPalindrome(string s) {
    if (s.empty())
      return string();
    int len = s.length();
    vector<vector<bool>> dp(len, vector<bool>(len));

    int maxLength = 0; // 用于记录最大子串的长度
    int startIndex = 0; // 记录最大子串的起始位置
    for (int i = 0; i < len; i++) {
      for (int j = i + 1; j < len; j++) {
        if (j - i <= 1)
          dp[i][j] = (s[i] == s[j]);
        else {
          dp[i][j] = (s[i] == s[j]) && dp[i + 1][j - 1];
        }

        // 不断记录子串的长度
        if (dp[i][j] && maxLength < j - i + 1) {
          maxLength = j - i + 1;
          startIndex = i;
        }
      }
    }
    // 返回最大子串
    return s.substr(startIndex, maxLength);
  }
};

/** 53. Maximum Subarray
 * @author: zlm
 * @date: 2018-6-25 22：12
 * @description: 求出一个数组中和最大的子数组（连续的）,返回subArray的和
 * @method: 这类题目比价常见, 直接遍历,O(n)
 * 1. 从第一个元素开始，记录当前累加和curSum, 如果curSum>0，则继续+下一个元素
 * 2. 若curSum < 0, 则即使加上下一个元素a[i]，结果肯定小于a[i]，故此时更新curSum = a[i],继续向后遍历
 */
class Solution53 {
public:
  int maxSubArray(vector<int>& nums) {
    if (nums.empty())
      return 0;

    int len = nums.size();
    int curSum = 0; // 当前累加的和
    int maxSum = INT_MIN;

    for (int i = 0; i < len; i++) {
      if (curSum < 0)
        curSum = nums[i];
      else {
        curSum += nums[i]; // 记录当前累加和
      }
      if (curSum > maxSum)
        maxSum = curSum;
    }
    return maxSum;
  }
};

/** 70. Climbing Stairs: 台阶问题
 * @author: zlm
 * @date: 2018-6-25 22：30
 * @description: 动态规划爬楼梯问题,给定n个台阶，每次限定只能走一阶或者两阶，求走台阶的方式个数
 * @method: 主要是递推公式的推导
 */
class Solution70 {
public:
  int climbStairs(int n) {
    if (n <= 1)
      return n;
    // 设f(i)表示爬i阶楼梯的方案个数，i：1~n
    // f(i) = f(i-1)+f(i-2)  最后一步可以走一阶也可以走两阶
    vector<int> f(n + 1);
    f[1] = 1;
    f[2] = 2;
    for (int i = 3; i <= n; i++) {
      f[i] = f[i - 1] + f[i - 2];
    }
    return f[n];
  }

  int climbStairs2(int n) {
    if (n <= 1) 
      return n;
    int f0 = 1, f1 = 0, f = 0;
    for (int i = 2; i <= n; i++) {
      f = f0 + f1;
      f0 = f1;
      f1 = f;
    }
    return f; 
  }
};

/** 128. Longest Consecutive Sequence
* @author: zlm
* @date: 2018-6-25 22：30
* @description: 数组中最长的连续子序列，返回子序列的元素个数
* @method: hash表实现。中心扩展法：以当前元素a[i]为中心，分别寻找右侧和左侧的相邻元素
*/
class Solution128 {
public:
  int longestConsecutive(vector<int>& nums) {
    if (nums.empty())
      return 0;

    unordered_map<int, bool> used; // 用来表示数组中的元素是否被用过
    for (auto v : nums)
      used[v] = false;

    int maxLength = 0;
    for (auto i : nums) {
      if (used[i])
        continue; // 如果元素已经使用过，则没必要再次使用

      int len = 1; // 记录当前查找的序列的长度
      used[i] = true;

      // 向右查找与元素i相邻的元素
      for (int j = i + 1; used.find(j) != used.end(); j++) {
        used[j] = true;
        len++;
      }

      // 向左查找与元素i相邻的元素
      for (int j = i - 1; used.find(j) != used.end(); j--) {
        used[j] = true;
        len++;
      }

      if (len > maxLength)
        maxLength = len;
    }
    return maxLength;
  }
};

/** 234. Palindrome Linked List
 * @author: zlm
 * @date: 2018-6-26 11：13
 * @description: 判断单链表是否是回文链表
 * @method: 要求空间复杂度为O(1)，时间复杂度为O(n)
 */
class Solution234 {
public:
  bool isPalindrome(ListNode* head) {
    if (head == nullptr || head->next == nullptr)
      return true;

    int len = 0;
    ListNode *tmp = head;
    while (tmp != nullptr) {
      len++;
      tmp = tmp->next;
    }

    // 找到链表中心
    tmp = head;
    for (int i = 1; i <= (len - 1) / 2; i++) {
      tmp = tmp->next;
    }
    // fast指向下标为(len-1)/2的结点

    ListNode *head2 = tmp->next; // head2 指向链表的后半段
    tmp->next = nullptr; // 前半段链表末尾置空
     
    // head2指向的链表翻转
    ListNode *revHead = Reverse(head2);

    // 前后两端链表对应元素比较
    // 如果是偶数，前后两段链表的长度均为n/2，如果是奇数，则不考虑中间元素
    // 前后比较的元素个数仍然为n/2个
    for (int i = 0; i < len / 2; i++) {
      if (head->val != revHead->val)
        return false;
      head = head->next;
      revHead = revHead->next;
    }
    return true;
  }

  // 翻转链表
  ListNode* Reverse(ListNode *head) {
    if (head == nullptr || head->next == nullptr)
      return head; // 只有一个结点，返回head自身

    ListNode *pcur = head->next;
    ListNode *pre = head;

    ListNode *pnext = nullptr;

    while (pcur != nullptr) {
      pnext = pcur->next;
      pcur->next = pre;

      pre = pcur;
      pcur = pnext;
    }
    // 此时pre指向翻转后的链表首结点
    head->next = nullptr;
    return pre;
  }
};

/** 647. Palindromic Substrings
 * @author: zlm
 * @date: 2018-6-26 15:02
 * @description: 计算字符串中含有多少个回文子串
 * @method: 比较高效的方式从每个字符开始，尝试扩展子字符串
 */
class Solution647 {
private:
  int cnt = 0;
public:
  int countSubstrings(string s) {
    if (s.empty())
      return 0;
    for (int i = 0; i < s.length(); i++) {
      HuiwenSubstring(s, i, i); // 奇数长度扩展
      HuiwenSubstring(s, i, i + 1); // 偶数长度扩展,如‘abba’
    }
    return cnt;
  }

  // 判断回文的函数
  void HuiwenSubstring(string &s, int left, int right) {
    while (left >= 0 && right < s.length() && s[left] == s[right]) {
      cnt++;
      left--; // 向左右扩展字符串
      right++;
    }
  }

};

/** 19. Remove Nth Node From End of List
 * @author: zlm
 * @date: 2018-6-26 15:02
 * @description: 从链表末尾删除第 N 个结点,并返回链表head
 * @method: 双指针：1. 从head开始，p1指针移动到第k个结点,此时剩余的结点个数为n-k
 * 2. 指针 p2 从 head 开始，p2 和 p1 同时向后移动，每次均移动一步；
 * 3. 当 p1 指向尾部结点时，p2 已经指向第 n-k+1 个结点, 即从后向前数第 k 个结点
 */
class Solution19 {
public:
  ListNode* removeNthFromEnd(ListNode* head, int n) {
    if (head == nullptr)
      return nullptr;

    ListNode *p1 = head;
    for (int i = 1; i <= n - 1 && p1 != nullptr; i++) { // p1 起始为第一个结点，向后移动 n-1 步
      p1 = p1->next;
    }
    if (p1 == nullptr) // 说明结点的个数小于 n
      return nullptr;

    // 否则说明p1此时指向正向第 n 个结点
    ListNode *p2 = head;
    ListNode *pre = nullptr; // 用于记录 p2 结点的前一个结点 
    while (p1->next != nullptr) { // p1向后移动 len-n 个结点
      p1 = p1->next;
      pre = p2;
      p2 = p2->next;
    }
    // 此时 p2 即为反向第 n 个结点, pre指向 p2 的前一个结点，反向第n+1个结点
    if (p2 == head) { // 如果待删除的结点为首结点head
      head = p2->next;
      delete p2;
      p2 = nullptr;
    }
    else {
      pre->next = p2->next;
      delete p2;
      p2 = nullptr;
    }
    return head;
  }
};

/** 136. Single Number, easy
 * @author: zlm
 * @date: 2018-6-26 15:02
 * @description: 找出只出现一次的数字
 * @method: 数组中只有一个数只出现一次，其余均出现两次，故可以使用异或来处理！
 */
class Solution136 {
public:
  int singleNumber(vector<int>& nums) {
    if (nums.empty())
      return 0;

    int res = 0; // 0^anyNum = anyNum
    for (int n : nums) {
      res = res ^ n;
    }
    return res;
  }
};

// 排列组合类题目
/** 46. Permutations
 * @author: zlm
 * @date: 2018-6-26 15:02
 * @description: 全排列问题
 * @method: 递归，先确定一个数，剩下的 n-1 个数数递归进行排列
 */
class Solution46 {
public:
  vector<vector<int>> permute(vector<int>& nums) {
    vector<vector<int>> res; // 存放结果
    if (nums.empty())
      return res;
    
    permutation(nums, 0, res);

    return res;
  }

  /* 核心函数
   * 参数：nums，数组；res：存放所有的排列；begin：当前排列时第一个元素的下标
   */
  void permutation(vector<int> &nums, int begin, vector<vector<int>> &res) {
    if (begin == nums.size()-1) { //如果遍历到最后一个元素，则把此次排列存入 res 中
      res.push_back(nums); // 因为这里是使用nums本身进行排列，故不需要return;
    }
    else {
      // 依次把后面元素跟第一个位置的元素交换
      for (int i = begin; i < nums.size(); i++) {
        swap(nums[begin], nums[i]);
        //递归的对 begin 后面的元素进行排列
        permutation(nums, begin + 1, res);

        // 排列后需要把原来的交换的元素再交换回来
        swap(nums[begin], nums[i]);
        // 开始下次循环，即使用下一个数作为首元素，再次进行排列
      }
    }  
  }

};

/** 77. Combinations
 * @author: zlm
 * @date: 2018-6-26 20:50
 * @description: 组合问题：在 1,2,3,..,n-1, n 中选出由 k 个数构成的所有组合
 * @method: backtracing， DFS， 递归
 * 先找出1开头的组合->2开头的组合->3开头的组合...需要对n个数进行DFS，故时间复杂度为 O(n!)
 */
class Solution77 {
public:
  vector<vector<int>> combine(int n, int k) {
    vector<vector<int>> res;

    if (n <= 0 || k > n)
      return res;

    vector<int> cur; // 当前组合
    combination(res, cur, n, k, 1); // 组合中的数字范围为：1~n

    return res;
  }
  // 组合的核心函数
  // begin：表示组合的起始元素。即求出从 begin 开始的元素
  void combination(vector<vector<int>> &res, vector<int> &cur, int n, int k, int begin) {
    if (cur.size() == k) { // 回溯的终止条件：找到一种组合后函数返回
      res.push_back(cur);
      return;
    }

    // 先确定首元素 i, 求出以 i 开头的组合
    for (int i = begin; i <= n; i++) {
      cur.push_back(i);

      // 递归
      combination(res, cur, n, k, i + 1); //确定了首元素 i, 然后对i之后的元素进行组合
          
      // 回溯后，删除当前for循环中添加的元素（line717），此时cur = [],下一次 fo r循环中将选择新的首元素
      cur.pop_back(); 
    }
  }
};

/** 617. Merge Two Binary Trees
* @author: zlm
* @date: 2018-6-27 Wed
* @description: 把两个二叉树对应结点元素相加，然后返回合并后的二叉树
* @method：递归实现：先合并root，然后对左子树和右子树递归进行合并
*/
class Solution617 {
public:
  TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
    if (t1 == nullptr && t2 != nullptr)
      return t2;
    else if (t2 == nullptr && t1 != nullptr)
      return t1;
    else if (t1 == nullptr && t2 == nullptr)
      return nullptr;

    TreeNode *root = new TreeNode(0);
    root->val = t1->val + t2->val; // 合并根结点

    // 递归对左子树和右子树进行合并
    root->left = mergeTrees(t1->left, t2->left);
    root->right = mergeTrees(t1->right, t2->right);

    return root;
  }
};

/** 113. Path Sum II
 * @author: zlm
 * @date: 2018-6-27 16:13
 * @description: 在二叉树中，找到所有结点和 =sum 的路径（这里的路径均为从 root结点->leaf结点的路径）
 * @method: 回溯，深度优先搜索 DFS
 */
class Solution113 {
public:
  vector<vector<int>> pathSum(TreeNode* root, int sum) {
    vector<vector<int>> res;
    if (root == nullptr)
      return res;
    
    vector<int> cur;
    FindPath(root, res, cur, sum);

    return res;
  }

  // 递归函数
  void FindPath(TreeNode *root, vector<vector<int>> &res, vector<int> &cur, int sum) {
    if (root == nullptr)
      return;

    cur.push_back(root->val);  // 先存入根结点
    
    if (root->left == nullptr && root->right == nullptr) { // 到达叶子结点后，当前路径已近确定，判断其路径和
      if (accumulate(cur.begin(), cur.end(), 0) == sum)
        res.push_back(cur);
    } else {
      if (root->left != nullptr) 
        FindPath(root->left, res, cur, sum);
      if (root->right != nullptr)
        FindPath(root->right, res, cur, sum);
    }

    // 回溯后，删除当前存入的结点
    cur.pop_back();
  }
};

//----------- Dynamic Programming examples -----------
/** 120. Triangle
 * @author: zlm
 * @date: 2018-7-11 16:43
 * @description: 三角形中，从顶点到最底层的最短路径，每次只能移动到下一层的相邻结点
 * @method: 动态规划__自上而下：
 * f[i,j] 表示从（0，0）点到点（i,j）的最短路径和
 * 到达（i,j）的前一个点只能是：（i-1,j-1）,(i-1,j)
 * 故f[i,j] = min(f[i-1,j], f[i-1,j-1]) + tri[i,j];
*/
class Solution120 {
public:
  int minimumTotal(vector<vector<int>>& triangle) {
    if (triangle.empty())
      return 0;

    int row = triangle.size();
    vector<vector<int>> dp(row, vector<int>(row)); // 默认初值为0

    // 动态表赋初值
    dp[0][0] = triangle[0][0]; // 第一行只有一个数

    for (int i = 1; i < row; i++) {
      for (int j = 0; j < row; j++) {
        if (j == 0) // 对于第一列，(i,j)的前一个落点只能是(i-1,j)
          dp[i][j] = dp[i - 1][j] + triangle[i][j];
        else if (j == triangle[i].size() - 1) // 当前行的最后一列
          dp[i][j] = dp[i - 1][j - 1] + triangle[i][j];
        else
          dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle[i][j];

      }
    }
  
    // dp[row-1][j]存放着从（0,0）到最后一层各点的最短路径
    int minimum = dp[row - 1][0];
    for (int i = 1; i < row; i++) {
      if (dp[row - 1][i] < minimum)
        minimum = dp[row - 1][i];
    }

    return minimum;
  }
};

/** 64. Minimum Path Sum
 * @author: zlm
 * @date: 2018-7-11 17:43
 * @description: 从矩阵的左上角到右下角的最小路径和，每次只能向左和向下移动。
 * @method: 动态规划
 */
class Solution64 {
public:
  int minPathSum(vector<vector<int>>& grid) {
    if (grid.empty() || grid[0].empty())
      return 0;
    int row = grid.size();
    int col = grid[0].size();

    vector<vector<int>> dp(row, vector<int>(col));

    // dp[0][0] = grid[0][0];
    for (int i = 0; i < row; i++) {
      for (int j = 0; j < col; j++) {
        if (i == 0 && j == 0)
          dp[i][j] = grid[i][j];
        else if (j == 0 && i >= 1) // 第一列
          dp[i][j] = dp[i - 1][j] + grid[i][j];
        else if (i == 0 && j >= 1) // 计算第一行
          dp[i][j] = dp[i][j - 1] + grid[i][j];
        else {
          dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
        }
      }
    }

    return dp[row - 1][col - 1];
  }  
};

/** 63. Unique Paths II: 带有障碍物，求路径个数
 * @author: zlm
 * @date: 2018-7-11 20:48
 * @description: 从矩阵的左上角到右下角的最小路径和，每次只能向左和向下移动。
 * @method: 动态规划
 */
class Solution63 {
public:
  int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
    // 边界条件
    if (obstacleGrid.empty() || obstacleGrid[0].empty())
      return 0;

    int row = obstacleGrid.size();
    int col = obstacleGrid[0].size();

    vector<vector<int>> dp(row, vector<int>(col));
    // 确定初值
    dp[0][0] = obstacleGrid[0][0] == 0 ? 1 : 0; // 有障碍时，路径个数为0

    // 第一列上的点只能向下走,注意障碍物
    for (int i = 1; i < row; i++) {
      dp[i][0] = (dp[i - 1][0] == 1 && obstacleGrid[i][0] == 0) ? 1 : 0; // 如果上一个点有路径，且当前点无障碍，则dp[i][0] = 1
    }

    // 第一行上的点只能向右走,注意障碍物
    for (int j = 1; j < col; j++) {
      dp[0][j] = (dp[0][j - 1] == 1 && obstacleGrid[0][j] == 0) ? 1 : 0; // 如果上一个点有路径，且当前点无障碍，则dp[0][j] = 1
    }

    // 其它点的递推公式
    for (int i = 1; i < row; i++) {
      for (int j = 1; j < col; j++) {
        if (obstacleGrid[i][j] == 1)
          dp[i][j] = 0;
        else
          dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
      }
    }
    return dp[row - 1][col - 1];
  }
};

/** 69. Sqrt(x): 求平方根
 * @author: zlm
 * @date: 2018-7-11 21:59
 * @description: xxx
 * @method: 二分查找法、牛顿迭代法
 */
class Solution69 {
public:
  int mySqrt(int x) {
    if (x <= 1)
      return x;

    int low = 1, high = x; // 对于大于1的正数，显然sqrt(x) < x
    int mid;
    while (low <= high) {
      mid = (low + high) / 2;
      if (mid * mid == x)
        return mid;
      else if (mid * mid < x)
        low = mid + 1;
      else high = mid - 1;
    }
    return high;
  }
};

// 牛顿迭代法： Xi+1 = Xi - F(Xi)/F'(Xi);
// 求x = sqrt(n), 相当于求x^2-n = 0, 令fx = x^2 - n ，可推导出 迭代公式 Xi+1 = (Xi + n/Xi)/2;
int sqrt2(int x) {
  if (x <= 1)
    return x;
  int s0 = 0, s1 = 1; //迭代初始值

  while (abs(s1 - s0) > 1.0e-6) {
    // 不满足精度，接着迭代求解
    s0 = s1;
    s1 = (s0 + x / s0) / 2;
  }
  return s0;
}

/** 135.Candy(贪心法实例)
 * date: 2018-7-13 19:14
 * author:zlm
 * 思路：贪心思想。
 * 首先每个小孩分一个糖果。设下标为i的孩子的糖果数量为num[i]，等级为r[i]。
 * 1. 从左到右扫描，第i个孩子等级 < 第i+1个孩子，num[i+1]=num[i]+1；
 * 2. 再从右向左扫描，第 i 个孩子等级 > 第i+1个孩子，同时第i个孩子的糖果数量 < 第i+1个孩子，则num[i]=num[i+1]+1;
 * 时间复杂度： 16ms，beats 98.51%
 */
class Solution135 {
public:
  int candy(vector<int>& ratings) {
    if (ratings.empty())
      return 0;

    int N = ratings.size();
    vector<int> candy(N, 1); // 每个人至少一块糖果

    // 从左向右扫描，找右侧等级高的孩子
    for (int i = 0; i < N-1; i++) {
      if (ratings[i] < ratings[i + 1])
        candy[i + 1] = candy[i] + 1;
    }

    // 从右向左扫描,找左侧等级高的孩子
    for (int i = N - 1; i >= 1; i--) {
      if (ratings[i - 1] > ratings[i] && candy[i - 1] <= candy[i])
        candy[i - 1] = candy[i] + 1;
    }

    // 返回candy之和
    int sum = 0;
    for (int i : candy)
      sum += i;

    return sum;
  }
};

/** 55. Jump Game 跳跃游戏, 8ms
 * @author: zlm
 * @date: 2018-7-13 20:43
 * @description: 给一个数组，里面每个元素表示可以向后跳跃的步数，我们需要知道能不能移动到最后一个元素位置。
 * @method: 贪心法，
 * 1. 假设当前可跳跃步数为 v，每次先跳跃一步，则剩余步数为 v-1，若此时新的位置的步数 s 大于 v-1，则使用新的步数继续移动;
 * 2. 由于剩余可跳跃次数 v-1 小于0,且没有到达数组末尾，说明当前已经无法再移动，即跳跃失败
 * 上述过程中，每跳跃到一个新的位置，都会选择最大的可用步数继续向前移动，是贪心思想的体现！
 */
class Solution55 {
public:
  bool canJump(vector<int>& nums) {
    // 向量为空，返回false
    if (nums.empty())
      return false;

    int len = nums.size();
    
    int step = nums[0]; // 从第一个元素开始跳跃

    for (int j = 1; j < len; j++) {
      step--; // 每次都先跳跃一步，然后比较剩余步数 (step-1) 与新位置的步数 nums[j]

      if (step < 0) // 说明可跳跃步数 < 1，即连一步都无法跳跃
        return false; 

      //跳跃到新位置 j 
      if (step < nums[j])
        step = nums[j]; // 在新的位置上选择更大的步数向后移动
    }

    return true;
  }
};

/** 45. Jump Game II 跳跃游戏：求最少跳跃次数
 * @author: zlm
 * @date: 2018-7-13 22:43
 * @description: 给一个数组，里面每个元素表示可以向后跳跃的步数，我们需要知道能不能移动到最后一个元素位置。
 * @method: 贪心策略：每一次跳跃的落点应该使得下一次跳跃到达的位置最远。 O(n), 空间O(1)
 * 1. 假设当前位置下标为 i，则一次可跳跃的最远位置为 i+A[i], 则在该区间(i, i+A[i])内的任意位置均可通过一次跳跃达到；
 * 2. 假设落点为 x，则应该使得 x+A[x]最大，因此下一个落点的区间为 (x, x+A[x]),之后重复上述过程；
 * 3. 这里每一次跳跃的落点都要求下一次能跳跃至最远位置
 */
class Solution45 {
public:
  int jump(int A[], int n) {
    if (n <= 1)
      return 0;

    int step = 0; // 最少的跳跃次数
    int cur = 0;  // 当前位置
    int furthest = 0; // 用来存放从当前位置一次跳跃可到达的最远位置

    // 从 cur 位置起跳
    while (cur < n) {
      step++;  // 当前区间内跳跃一次
      furthest = cur + A[cur];
      if (furthest >= n - 1)
        return step;
      // 确定区间(cur, furthest]内的落点
      int tmp = -1;
      int index = 0;
      for (int i = cur + 1; i <= furthest; i++) {
        if (tmp < A[i] + i) {
          tmp = A[i] + i;
          index = i; // 记录当前位置
        }
      } // end for，此时index为区间内的落点
      cur = index;
    }
    return -1; // 如果while内部没有返回step，则说明无法跳跃至终点
  }
};

/** 83. Remove Duplicates from Sorted List(与剑指offer题目不完全相同)
 * @author: zlm
 * @date: 2018-7-14 23:22
 * @description: 删除链表中的重复元素，只保留一个, 例如，链表1->2->3->3->4->4->5 处理后为 1->2->3-4-5
 * @method: 双指针，pre 和 pcur；创建一个新的头结点newHead，依次把链表中非重复结点添加到 newHead 后面
 */
class Solution83 {
public:
  ListNode* deleteDuplicates(ListNode* head) {
    if (head == nullptr)
      return nullptr;

    ListNode *newHead = new ListNode(0);
    newHead->next = head;

    ListNode *pcur = head;
    ListNode *pre = newHead; 

    while (pcur != nullptr && pcur->next != nullptr) {
      if (pcur->val == pcur->next->val) { // 若 pcur 指向重复元素, 则需要找到与该元素不重复的下一个元素
        int tmp = pcur->val;

        while (pcur != nullptr && pcur->val == tmp) {
          pcur = pcur->next;
        }

        // pre是指向pcur前面非重复的结点，执行下面语句可把pre和pcur之间的重复结点删除
        pre->next = pcur;
        // pre = pcur, 加上此句可以保留一个重复元素，否则，删除所有重复元素
      }
      else { // pcur->val != pcur->next->val
        pcur = pcur->next;
        pre = pre->next; // pre 和 pcur 均向右移动
      }
    } // end while
    
    return newHead->next;
  }
};

/** 3. Longest Substring Without Repeating Characters
 * @author: zlm
 * @date: 2018-7-15 11:43 第二次练习
 * @method: 使用 STL 几何 set 存放字符。left,right 均为字符串 s 的下标。
 * 依次把字符s[right]添加至集合 set 中。遇到重复元素，则把集合中已经存在的元素及其之前的元素删除
 */
class Solution3 {
public:
  int lengthOfLongestSubstring(string s) {
    int len = s.length();
    set<char> st;
    int maxLen = 0;
    int left = 0,right = 0;
  
    while (right < len && left < len) {
      if (st.find(s[right]) == st.end()) // 若s[right]在集合 st 中没有出现过，则将其加入集合
      {
        st.insert(s[right]);
        right++;
        maxLen = max(maxLen, right - left); // right-left即为当前集合中元素的个数
      }
      else { // 说明s[right]在st中已经出现过，则把结合中该重复元素及其之前的元素删除
        st.erase(s[left]);
        left++;
      }
    } 

    return maxLen;
  }
};

/**61. Rotate List: 要求把后面k个节点翻转到链表前面,如1-2-3-4-5-null, k = 2, res = 4-5-1-2-3-null
 * @author: zlm
 * @date: 2018-7-15 16:26
 * @method: 设链表长度为len,链表分为两部分：前 n-k 和节点和后 k 个节点(若k>len，取 k % len)。
 * 1. 遍历一次，求出len, 由于后k个节点翻转到前面，因此最后一个结点一定与第一个结点首尾相连； 
 * 2. 遍历前n-k个节点，断开第n-k个节点与下一个结点的连接
 * @复杂度：时间O(),空间O()
 */
class Solution61 {
public:
  ListNode* rotateRight(ListNode* head, int k) {
    if (head == nullptr || k <= 0)
      return nullptr;

    // 1. 遍历求链表长度
    int len = 0;
    ListNode *pcur = head;
    ListNode *pre = head;
    while (pcur != nullptr) {
      len++;
      pre = pcur;
      pcur = pcur->next;
    }
    pre->next = head; // 首尾连接

    // 2. 遍历前 len-k%len个结点,并断开连接
    pcur = head; 
    for (int i = 1; i < len - k%len; i++) {
      pcur = pcur->next;
    } // pcur指向第len-k%len个结点，其下一个结点为反向第k个结点

    ListNode *newHead = pcur->next;
    pcur->next = nullptr;

    return newHead;
  }
};

/**剑指offer:最小的 K 个数
 * @author: zlm
 * @date: 2018-7-16 20：10
 * @description: 统计数组中最小的 k 个数
 * 方法 2: 大顶堆。使用前K个元素构造一个大顶堆，然后依次把后面的元素与堆顶对比，
 * 若小于堆顶，则替换之，并重新调整堆；
 * 最终堆中元素为最小的前k个元素
 * 这里使用 优先队列（默认为大顶堆，队首元素最大（top()返回队首），队尾元素最小）
 */
class Solution112 {
public:
  vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
    vector<int> res;
    if (k <= 0 || k > input.size())
      return res;
    priority_queue<int> que;

    for (int i = 0; i < input.size(); i++) {
      que.push(input[i]);
      if (que.size() == k + 1) // 前k+1个元素已经排好序（从大到小），其中队首最大，剩下的k个元素最小
        que.pop(); // 出队，即删除队首
    }

    while (!que.empty()) {
      res.push_back(que.top());  // 依次出队
      que.pop();
    }
    return res;
  }
};

/** 调整数组顺序使奇数位于偶数的前面
 * @author: zlm
 * @date: 2018-7-18 22:10
 * @method1: 使用两个向量，从前向后遍历数组，奇数放入vec1，偶数放入vec2，然后将两个向量的元素合并至array
 * @method1: 使用 STL 双端队列，偶数通过push_back放在队列的后面；奇数通过push_front放在前面（为保证奇数
 * 之间相对次序的不变，应该从数组尾部向前扫描奇数！！！）
 * @复杂度：时间O(n),空间O(n)
 */
class Solution111 {
public:
  void reOrderArray(vector<int> &array) {
    if (array.empty())
      return;

    deque<int> dq;

    int len = array.size();

    for (int i = 0; i < len; i++) {
      if (array[i] % 2 == 0) // 偶数放在dq的后面
        dq.push_back(array[i]);
      // 对于奇数，通过push_front，应该从数组末尾向前扫描，保证在队列中相对位置不变
      if (array[len - i - 1] % 2 == 1)
        dq.push_front(array[len - 1 - i]);
    }
    array.assign(dq.begin(), dq.end()); // assign函数用于赋值，可以用于不同容器但同类型元素的赋值
  }
};

/** 剑指offer题，把数组排列成最小的数
 * @description: 例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
 * @author: zlm
 * @date: 2018-7-19 10:18
 * @method1:把数字转换成字符串
 */
class Solution114 {
public:
  string PrintMinNumber(vector<int> numbers) {
    if (numbers.empty())
      return "";

    // 1 数字转换成string
    int len = numbers.size();
    vector<string> strNum(len);
    for (int i = 0; i < len; i++)
    {
      strNum[i] = to_string(numbers[i]);
    }
    // 2 按自定义规则排序
    sort(strNum.begin(), strNum.end(), myCompare);

    // 3 拼接字符串
    string res;
    for (int i = 0; i < len; i++)
      res += strNum[i];

    return res;
  }
  // 使用 sort 算法进行排序式，需要自己定义排序规则，使得排序的 strNum 满足题目要求
  // 对于两个字符串 A，B
  // 若 AB < BA, 则应该把A放在前面，B放在后面，即认为 A“小于”B
  // 若 AB > BA, 则应该把B放在前面, 认为 B "小于" A。
private:
  static bool myCompare(const string &a, const string &b) { // 在类中自定义sort的比较函数，必须是static的
    return a + b < b + a;
  }
};

/**
 * @description: 剑指offer题，丑数
 * @author: zlm
 * @date: 2018-7-19 11:23
 * @method 2: 对于丑数num，只能被2、3、5整除，则num除以2或者3、5（如果能除尽）的结果显然也是丑数
 * 即丑数num = pre * (2||3||5)，pre 为小于 num 的一个丑数
 */
class Solution116 {
public:
  int GetUglyNumber_Solutiion(int index) {
    if (index < 1)
      return 0;

    vector<int> pre; // 用来存放已有的丑数(其中按顺序存放)
    pre.push_back(1); // 1是第一个丑数
    int i2 = 0, i3 = 0, i5 = 0;
    // 下一个丑数 next = pre * 2||3||5，im记录的是乘以 m 的前一个丑数pre
    int tmp;

    while (pre.size() < index) {
      tmp = min(pre[i2] * 2, min(pre[i3] * 3, pre[i5] * 5)); 
      pre.push_back(tmp);

      // 找到是pre[i2]还是 pre[i3] 或者 pre[i5]，若tmp = pre[im]*m,则im之前的元素*2肯定小于tmp，故不需要再考虑
      // 此时令im+1，其他的 im 不变（因为 pre[ik]*k 大于 tmp，k!=m，ik*k的值有可能是tmp后面的丑数，故ik不需要移动）;
      // 下一次循环再次找出 pre[im]*m 的最小值做为下一个一个丑数
      if (pre[i2] * 2 == tmp)
        i2++;
      if (pre[i3] * 3 == tmp)
        i3++;
      if (pre[i5] * 5 == tmp)
        i5++;
    }
    // pre.size()==index
    return pre[index - 1];
  }
};

/**
 * @description: 剑指offer题目，构建乘积数组
 * @author: zlm
 * @date: 2018-7-19 15:03
 * @method:要求不能用除法。先从前向后扫描，计算B[i]=A0*A1*...*Ai-1，再从后向前计算
 * B[n-1] = A0***An-2*An(An用1表示)
 * B[n-2] = A0***An-3*An-1
 */
class Solution_117 {
public:
  vector<int> multiply(const vector<int>& A) {
  
    int len = A.size();
    if (len < 1)
      return vector<int>();

    vector<int> B(len);
    B[0] = 1;
    int tmp = A[0];
    for (int i = 1; i < len; i++) {
      B[i] = tmp;
      tmp *= A[i];
    }
    // 从后向前累乘
    tmp = 1;
    for (int i = len - 1; i >= 0; i--) {
      B[i] *= tmp;
      tmp *= A[i];
    }
    return B;
  }
};

/**
 * @description: 数组中的逆序对
 * @author: zlm
 * @date: 2018-7-19 15:59
 * @method: 归并排序的思想，在每次合并过程中累计逆序对的个数
 * 通过递归调用归并函数不断的划分子序列，最终得到 n 个长度为 1 的有序子序列
 * 对有序子序列两两合并，合并过程中需要计算两个序列所构成的逆序对的个数 cnt，合并后形成
 * 的子序列是有序的，这是防止后续的合并过程重复计算 cnt
 */
class Solution_51 {
private:
  vector<int> tmp; // 用于存放合并后的元素，避免在merge函数中递归的创建
  int cnt = 0; // C++11运行类内初始化
public:
  int InversePairs(vector<int> data) {
    if (data.size() <= 1)
      return 0;

    tmp.resize(data.size());
    int low = 0, high = data.size() - 1;
    // cnt的累计在 merge 函数中进行
    mergeSort(data, low, high);

    return (cnt % 1000000007);
  }
  // 归并排序相关的函数
  void mergeSort(vector<int> &data, int low, int high) {
    if (low < high) {
      int m = (low + high) / 2; //把序列分成两段，递归调用mergeSort
      mergeSort(data, low, m);
      mergeSort(data, m, high);

      //至此已经实现a[low~m]和a[m+1~high]有序
      merge(data, low, m, high);
    }
  }
  // 在合并子序列的过程中，统计逆序对的个数
  void merge(vector<int> &a, int low, int mid, int high) {
    int i = low, j = mid+1, k = low;
    while (i <= mid && j <= high) {
      if (a[i] <= a[j])
        tmp[k++] = a[i++];
      else {
        tmp[k++] = a[j++];  // a[i] > a[j]，说明a[i...mid] 都大于 a[j]
        this->cnt += mid - i + 1;
      }    
    }
    while (i <= mid)
      tmp[k++] = a[i++];
    while (j <= high)
      tmp[k++] = a[j++];

    // 把合并后的结果复制到原数组
    for (k = low; k <= high; k++)
      a[k] = tmp[k];
  }
};

/**
 * @description: 30 包含min函数的栈
 * @author: zlm
 * @date: 2018-7-19 15:59
 * @method: 使用两个栈，数据栈 data 和最小栈 Min, Min栈顶记为 minimum。若新元素value小于minimum，则两个栈均入栈；
 * 若value>minimu, 则把value压人到data栈，Min不压入。
 */
class Solution_min_Stack {
private:
  stack<int> data, Min;
  // data 栈用于存放数据，min 的栈顶用于存放最小元素
  int minimum;
public:
  void push(int value) {
    if (data.empty() && Min.empty()) {
      data.push(value);
      Min.push(value);
      minimum = value;
    }
    else {
      if (value <= minimum) { // 
        data.push(value);
        Min.push(value);
        minimum = value;
      }
      else {
        data.push(value); 
      }
    }
  } // end push
  
  void pop() {
    if (!data.empty()) {
      if (data.top() == Min.top()) {
        data.pop();
        Min.pop();
      } else {
        data.pop();
      }  
    }
  }
  
  int top() {
    if (data.empty())
      return -1;
    return data.top();
  }

  int min() {
    if (Min.empty())
      return -1;
    return Min.top();
  }
};

/**
 * @description: 53 表示数值的字符串
 * @author: zlm
 * @date: 2018-7-19 23:00
 * @method: 直接遍历
 */
class Solution_numeric_str {
public:
  bool isNumeric(char* string) {
    if (string == nullptr || strlen(string))
      return false;

    int index = 0;
    int len = strlen(string);

    // 1. 判断符号
    if (string[index] == '+' || string[index] == '-')
      index++;

    //若只有符号位，则不是正确的数字
    if (index == len) 
      return false;

    // 2. 扫描数字部分
    index = scanDigits(string, index);
    if (index < strlen(string)) {
      // 3. 如果有小数
      if (string[index] == '.') {
        index++;
        index = scanDigits(string, index);
        // 4. 如果有指数表示的形式
        if (index < strlen(string)) {
          if (string[index] == 'e' || string[index] == 'E') {
            index++;
            return isExponential(string, index);
          }
          return false;
        }
        return true;
      }
      else if (string[index] == 'e' || string[index] == 'E') {
        //如果没有小数，且有指数形式
        index++;
        return isExponential(string, index);
      }
      return false;
    }
    return true;
  }

  bool isExponential(char *str, int index) {
    if (index < strlen(str)) {

      //如果是符号，移动至下一个
      if (str[index] == '+' || str[index] == '-') {
        index++;
      }

      index = scanDigits(str, index);

      if (index == strlen(str)) 
        return true;

      return false;
    }
    return false;
  }
  int scanDigits(char *str, int index) {
    while (index < strlen(str) && str[index] >= '0' && str[index] <= '9') 
      index++;
    return index;
  }
};

/**
 * @description: 239. Sliding Window Maximum(滑动窗口最大值)
 * @author: zlm
 * @date: 2018-7-20 9:30
 * @method 2 : 利用双端队列，通过一系列入队和出队操作计算滑动窗口最大值
 * 1. 数组元素从i=0开始入队，若新入队的元素大于前一个元素（即当前队尾），则把队尾元素剔除，因为此时队尾元素不可能为窗口的最大值；
 * 2. 若新入队元素小于队尾，则保留队尾，同时新元素入队（因为随着窗口的移动，该元素有可能成为某个窗口最大值）
 * 3. 随着新元素入队，要及时判断队头元素是否在当前窗口内：通过下标差是否>=k来判断，若不在窗口内，则将队头出列；否则暂时保留！
 * 4. 通过上述操作，队头始终为当前窗口的最大值。AC, 44ms
 */
class Solution239 {
public:
  vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    // border cases 边界情况
    if (nums.empty() || k > nums.size() || k <= 0)
      return vector<int>();

    deque<int> dq; // 队列中存放的下标

    vector<int> res;

    int n = nums.size();
    // 先将前 size 个元素入队：这是第一个窗口
    // 举例：先入队2 3 4， 队列中的元素为：4(2,3小于4，被剔除)

    int i = 0;
    for (; i < k; i++) {
      //新入队元素大于队尾，剔除队尾
      while (!dq.empty() && nums[i] >= nums[dq.back()])
        dq.pop_back();

      dq.push_back(i);
    } // 第一个窗口元素经过入队操作后，在队列中只剩下最大元素 4

    res.push_back(nums[dq.front()]);

    // 对于下标i=size~n-1的元素，每个元素都是对应窗口的最后一个元素
    // 除第一个窗口，剩下窗口个数为 n-size
    for (int i = k; i < n; i++) {
      // 首先判断当前队头是否处在当前窗口内
      if (i - dq.front() >= k)
        dq.pop_front();
      while (!dq.empty() && nums[i] >= nums[dq.back()])
        dq.pop_back();
      dq.push_back(i);

      // 队头为当前窗口最大值
      res.push_back(nums[dq.front()]);
    } //end for

    return res;
  }
};

/**
 * @description: 63. 剑指offer,数据流中的中位数
 * @author: zlm
 * @date: 2018-7-20 9:30
 * @method: 使用大顶堆和小顶堆，且小顶堆的堆顶>=大顶堆堆顶，则小顶堆的所有元素均>=大顶堆;
 * 如果数据个数为奇数，则小顶堆元素个数比大顶堆多出一个，其堆顶即为median;
 * 如果数据为偶数个，则两个堆元素个数相同，median = 两堆顶元素之和/2;
 * 算法实现过程：1. 数据流读取过程中，将数据分摊到两个堆上（奇数位次的数放在小顶堆，偶数位次的数放在大顶堆）;
 * 2. 设当前读取的数为数据第 m 个数, 若 m % 2 = 1, 则先把 m 压入 big_heap, 
 * 然后再把 big_heap 的堆顶元素取出, 放入 small_heap
 * 这是为保证小顶堆的元素始终大于大顶堆！
 * 3. 同样若 m % 2 = 0, 则先把 m 压入 small_heap, 再把 small_heap 的堆顶取出放到 big_heap 中。
 * 上述过程可始终保证小顶堆的堆顶 >= 大顶堆的堆顶。
 */
class Solution_63 {
private:
  priority_queue<int, vector<int>, less<int>> maxHeap; // 默认大顶堆，即从大到小排序，队首top最大
  priority_queue<int, vector<int>, greater<int>> minHeap; // 小顶堆(top最小)
  int cnt = 0; // C++11里支持这种类内初始化
public:
  void Insert(int num) {
    cnt++; // 统计当前读取的数据流个数
    int tmp;

    if (cnt % 2 == 0) { // 偶数位次放入到大顶堆中(需要先压入小顶堆，取出栈顶再放入大顶堆)
      minHeap.push(num);
      maxHeap.push(minHeap.top());

      minHeap.pop();
    }
    else if (cnt % 2 == 1) {
      maxHeap.push(num);
      minHeap.push(maxHeap.top());

      maxHeap.pop();
    }

  }

  double GetMedian() {
    if (cnt % 2 == 0)
      return (minHeap.top() + maxHeap.top()) / 2.0;
    else
      return minHeap.top();
  }

};

/**
 * @description: 66. 剑指offer,机器人的运动范围
 * @author: zlm
 * @date: 2018-7-20 17:31
 * @method：回溯法, 递归
 */
class Solution_64 {
public:
  int movingCount(int threshold, int rows, int cols) {
    
    vector<vector<bool>> isVisited(rows, vector<bool>(cols, false)); // 标记网格是否被访问过

    int cnt = 0;

    if (threshold < 0 || rows <= 0 || cols <= 0) {
      return cnt;
    }
    
    cnt = backTracking(threshold, 0, 0, rows, cols, isVisited);

    return cnt;
  }

  // 回溯的核心函数
  // 这里的 row 和 col表示当前的坐标
  int backTracking(int threshold, int r, int c, int rows, int cols, vector<vector<bool>> &isVisited) {
    int cnt = 0;
    // 回溯的条件
    if (!check(threshold, r, c, rows, cols)) // 坐标(r, c)无法进入返回 0
      return 0;

    // 若可以进入(row, col), 则递归的计算从下一个位置出发经过的格子数
    // 下一位置：(r,c-1) (r, c+1) (r-1, c) (r+1, c)
     if (!isVisited[r][c]) {
      isVisited[r][c] = true;
      cnt = 1 + backTracking(threshold, r, c - 1, rows, cols, isVisited) \
        + backTracking(threshold, r, c + 1, rows, cols, isVisited) \
        + backTracking(threshold, r - 1, c, rows, cols, isVisited) \
        + backTracking(threshold, r + 1, c, rows, cols, isVisited);
    }
    return cnt;

  }

  // 检查当前位置(i,j)能否进入
  bool check(int threshold, int r, int c, int rows, int cols) {
    bool flag = (r >= 0 && r < rows) && (c >= 0 && c < cols);
    if (flag && (sumOfDigit(r) + sumOfDigit(c) <= threshold) )
      return true;
    else
      return false;
  }

  // 计算数字各个位之和
  int sumOfDigit(int d) {
    int sum = 0;
    while (d != 0) {
      sum += d % 10;
      d = d / 10;
    }
    return sum;
  }
};

/**
 * @description: 65. 矩阵中的路径
 * @author: zlm
 * @date: 2018-7-20 19:01
 * @method：回溯法, 递归
 * 1. 从矩阵 matrix 中任选一个字符作为 c0 起点，给定路径上的字符为 str[i];
 * 2. 若 str[i] != c0，表明 c0 不处在路径上的第 i 个位置; 若 c0 == str[i], 表明
 * c0 对应给定路径上的第 i 个字符,则在与 c0 相邻的节点上寻找路径上的第 i+1 个字符。
 * 3. 除边界上的字符，其余字符均有4个相邻的节点(上、下、左、右)。
 * 步骤 2 中会递归调用函数实现路径的查找。若 str[] 中所有的字符均在 matrix 中找到对应节点，则查找成功;
 * 否则查找失败
 */
class Solution_65 {
public:
  // str是给定的路径
  bool hasPath(char* matrix, int rows, int cols, char* str)
  {
    vector<vector<bool>> isVisited(rows, vector<bool>(cols, false)); // 标记矩阵中的节点是否被访问过

    // 任选一个结点作为起点(矩阵中每个点都有可能是起点，故需要对每个点进行判断)
    int index = 0;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        if (hasPathBackTracking(matrix, i, j, rows, cols, str, index, isVisited))
          return true;
      }
    }
    return false;
  }

  // 用于判断从某点出发是否能得到包含给定路径的一条路径！
  // r,c 路径起点, index：str中字符的位置
  bool hasPathBackTracking(char *matrix, int r, int c, int rows, int cols, char *str, int &index, vector<vector<bool>> &isVisited)
  {
    // 回溯条件，当str[]中最后一个字符在 matrix 中找到对应位置时，查找成功
    if (index == strlen(str))
      return true;
    
    bool pathFinded = false; // 是否找到路径的标志位

    if (check(matrix, r, c, rows, cols, str, index, isVisited)) {
      index++;
      isVisited[r][c] = true;

      // 递归的从[r,c]的相邻结点寻找路径
      // 下一位置：(r,c-1) (r, c+1) (r-1, c) (r+1, c)
      pathFinded = hasPathBackTracking(matrix, r, c - 1, rows, cols, str, index, isVisited) \
        || hasPathBackTracking(matrix, r, c + 1, rows, cols, str, index, isVisited) \
        || hasPathBackTracking(matrix, r - 1, c, rows, cols, str, index, isVisited) \
        || hasPathBackTracking(matrix, r + 1, c, rows, cols, str, index, isVisited);

      // 若pathFinded == fales,说明当前结点(r,c)的相邻结点均无法匹配路径str中的第index+1个字符，此时
      // 回到前一个字符（令index--），同时标记isVisited[r][c]=false, 即寻找其他的节点与路径str[i]进行匹配
      if (!pathFinded) {
        index--;
        isVisited[r][c] = false;
      }
    }
    return pathFinded;    
  }

  // 检测是否继续寻找当前矩阵结点 (r,c) 相邻的节点
  bool check(char *matrix, int r, int c, int rows, int cols, char *str, int &index, vector<vector<bool>> &isVisited) {
    bool flag = (r >= 0 && r < rows) && (c >= 0 && c < cols) && (str[index] == matrix[r*cols + c]) \
      && !isVisited[r][c];

    return flag;
  }
};

/**
 * @description:
 * @author:zlm
 * @date:2018-7-20 21:56
 * @method: B 是 A 的子结构，有以下几种情况：
 * 1. B 与 A 的根结点相同（此时 B 的子树的根必然与 A 的子树的根结点相同）
 * 2. B 是 A 的左子树的子结构，不考虑 A 的根节点
 * 3. B 是 A 的右子树的子结构，不考虑 A 的根节点
 * 因此针对 2 和 3，可递归处理！对于情况1，需要单独写一个函数进行判断
 */
class Solution_subtree {
public:
  bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2) {
    if (pRoot1 == nullptr || pRoot2 == nullptr)
      return false;
    return isSubtreeWithSameRoot(pRoot1, pRoot2) \
      || HasSubtree(pRoot1->left, pRoot2) \
      || HasSubtree(pRoot1->right, pRoot2);
  }
  
  bool isSubtreeWithSameRoot(TreeNode* p1, TreeNode* p2) {
    if (p2 == nullptr) // p2树中的结点判断完毕
      return true; 
    if (p1 == nullptr)
      return false;

    // p1和p2均不为空
    if (p1->val != p2->val)
      return false;
    else
      return isSubtreeWithSameRoot(p1->left, p2->left) && isSubtreeWithSameRoot(p1->right, p2->right);
  }
};

/**
 * @description:
 * @author:zlm
 * @date:2018-7-21 14:07
 * @method: 中序，递归
 * 1. 按照中序遍历顺序，当遍历到root时，他的左子树已经形成了排序链表，将此链表与root连接;
 * 2. 之后，递归处理右子树使其形成链表，再与root连接。
 */
class Solution_bst_to_LinkedList {
private:
  TreeNode *pLastNode = nullptr; // 已经排序好的链表中的最后一个结点指针
  TreeNode *head = nullptr;
public:
  TreeNode* Convert(TreeNode* pRootOfTree)
  {
    if (pRootOfTree == nullptr)
      return nullptr;
    InorderConvert(pRootOfTree);
    return head;
  }

  // 采用中序遍历的思路进行转换
  void InorderConvert(TreeNode* root) {
    if (root == nullptr)
      return; // 递归终止条件

    // 1. 先对左子树递归处理，形成排序链表
    InorderConvert(root->left);

    // 2. 将左子树链表最后一个结点 pLastNode 与当前 root 连接
    root->left = pLastNode;
    if (pLastNode == nullptr)
      pLastNode = root; // 当 pLastNode 为空时，已经遍历至最左边的结点，即链表的第一个结点
    else { // pLastNode!=null，指向已排序链表的最后一个结点
      pLastNode->right = root; // 链表中增加了一个结点root,故pLastNode向后移动一位
      pLastNode = root;
    }
    if (head == nullptr) // 此时遍历至树的最左边结点，将其作为头结点
      head = root;
    // 3. 经过 1 和 2, 所遍历的结点均已形成链表，且 pLastNode指向链表的最后一个结点，
    // 此时递归的对右子树进行处理
    InorderConvert(root->right);
  }
};

/**
 * @description: 删除指定的单链表的一个节点，要求时间复杂度为O(1)
 * @author:zlm
 * @date:2018-7-29 14：39
 * @method: 把指定结点的后继节点的值赋给当前结点。这样无需要查找待删除节点的前驱结点
 */
void DeleteNode(ListNode* &pListHead, ListNode *node) {
  if (pListHead == nullptr || node == nullptr)
    return;

  if (node->next != nullptr) {
    ListNode *pNext = node->next; // node在前，pNext在后
    node->val = pNext->val;

    node->next = pNext->next;
    // 删除pNext
    delete pNext;
    pNext = nullptr;
  }
  else if (node == pListHead) { // 删除的是头结点
    delete node;
    pListHead = node = nullptr;
  } 
  else { // 待删除节点为尾节点
    ListNode *pTemp = pListHead;
    while (pTemp->next != node)
      pTemp = pTemp->next;

    pTemp->next = nullptr;
    delete node;
    node = nullptr;
  }
}

/**
 * @description: 句子翻转（leetcode）
 * @author:zlm
 * @date:2018-10-4 14:20
 * @method: 先整体翻转，在对单词进行翻转，时空复杂度要求O(N)和O(1)
 * 单词之间只能保留一个空格，句子前后的空格需要删除！
 */
void reverseWords(string &s) {
    if (s.size() < 1)
        return;

    // 整体反转
    reverse(s.begin(), s.end());
    
    int start = 0, j = 0; // s[start-j]用来标记一个单词
    int len = s.size();
            
    while (j < len) {
        // 遇到单词前的空格，需要删除之
        if (s[j] == ' ') {
            if (j != start) { // 此时说明，start-j之间存在单词
                reverse(s.begin()+start, s.begin()+j);
                start = ++j; // start从下一个单词的首字母开始
            }
            else { // j == start,表明此处为空格，应删除之
                // 删除从第 j 位置开始的一个字符，此时 len = len-1, j 不变，s中被删除字符后面的字符向前移动了一位。
                s.erase(j, 1);
                --len;
            }
        }  
        // s[j] != ' '，j 继续移动
        else 
            ++j;
    }

    // 若句子最后有空格，则退出 while 循环时，start = j = len;
    // 若句子最后没有空格，则最后一个单词没有还没有实现翻转。
    reverse(s.begin()+start, s.begin()+j);
        
    // 去除句子最右边的空格
    if (s.back() == ' ')
        s.pop_back();
}

// test codes
int main()
{
  //priority_queue<int> q;
  //for (int i = 10; i > 0; i--)  
  //  q.push(i); // 默认按从大到小排列，队首最大（q.top指向队首）

  //cout << q.top() << endl;
  //q.pop();

  //cout << q.top() << endl;
  //q.pop();

  //cout << q.top() << endl;
  //q.pop();

  //q.push(100);
  //cout << q.top() << endl;

  // 使用优先队列构造小顶堆：元素按从小到大排列
  //priority_queue<int, vector<int>, greater<int>> q; // 需要包含头文件<functional>

  int i = 0;
  int a = i++;
  cout << a << " " << i << endl;
  return 0;
}
