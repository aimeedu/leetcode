
// One leetcode a day, keep unemployment away.

// 268. Missing Number
class Solution {
    public int missingNumber(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < nums.length; i++) {
            n = n ^ i ^ nums[i];
        }
        return n;
    }
}

class Solution {
    public int missingNumber(int[] nums) {
        Set<Integer> numSet = new HashSet<Integer>();
        for (int n : nums)
            numSet.add(n);
        // if no number is missing, total will be nums.length+1
        int total = nums.length + 1;
        for (int n = 0; n < total; n++) {
            if (!numSet.contains(n)) {
                return n;
            }
        }
        return -1;
    }
}

// 771. Jewels and Stones
class Solution {
    public int numJewelsInStones(String J, String S) {
        int jew_count = 0;
        for (int i = 0; i < S.length(); i++) {
            if (J.indexOf(S.charAt(i)) != -1) {
                jew_count++;
            }
        }
        return jew_count;
    }
}

// 169. Majority Element
// Approach 1: Brute Force
class Solution {
    public int majorityElement(int[] nums) {
        int majorityCount = nums.length / 2;
        for (int num : nums) {
            int count = 0;
            for (int ele : nums) {
                if (ele == num) {
                    count++;
                }
            }
            if (count > majorityCount) {
                return num;
            }
        }
        return -1;
    }
}

// 01/08/2020

// 49. Group Anagrams
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        // ArrayList<String> final = new ArrayList<>();
        Map<String, ArrayList<String>> map = new HashMap<>();
        for (String s : strs) {
            char[] sortedChar = s.toCharArray();
            Arrays.sort(sortedChar);
            String sorted = new String(sortedChar);
            if (map.get(sorted) == null) {
                ArrayList<String> l = new ArrayList<>();
                l.add(s);
                map.put(sorted, l);
            } else {
                map.get(sorted).add(s);
            }
        }
        // System.out.println(map.values() instanceof Collection<>);
        return new ArrayList(map.values());

    }
}

// 215. Kth Largest Element in an Array
class Solution {
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> q = new PriorityQueue<>(k + 1);
        for (int n : nums) {
            q.add(n);
            if (q.size() > k) {
                q.poll();
            }
        }
        return q.peek();
    }
}

// 703. Kth Largest Element in a Stream
class KthLargest {
    int k;
    ArrayList<Integer> numsList;

    public KthLargest(int k, int[] nums) {
        this.k = k;
        numsList = new ArrayList<>();
        for (int i : nums)
            numsList.add(i);
    }

    public int add(int val) {
        numsList.add(val);
        Collections.sort(numsList);
        return numsList.get(numsList.size() - k);
    }
}

// 94. Binary Tree Inorder Traversal
class Solution {
    // declare arraylist must be global, or we need to use a helper function.
    List<Integer> l = new ArrayList<>();

    public List<Integer> inorderTraversal(TreeNode root) {
        if (root == null)
            return l;
        inorderTraversal(root.left);
        l.add(root.val);
        inorderTraversal(root.right);
        return l;
    }
}

// 01/10/2020

// 102. Binary Tree Level Order Traversal

/**
 * Definition for a binary tree node. public class TreeNode { int val; TreeNode
 * left; TreeNode right; TreeNode(int x) { val = x; } }
 */
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        Queue<TreeNode> q = new LinkedList<>();
        List<List<Integer>> ans = new ArrayList<>();
        if (root == null) {
            return ans;
        }
        q.add(root);
        while (!q.isEmpty()) {
            List<Integer> t = new ArrayList<>();
            // q.add(root);
            int size = q.size();
            for (int i = 0; i < size; i++) {
                TreeNode c = q.remove();
                t.add(c.val);
                if (c.left != null)
                    q.add(c.left);
                if (c.right != null)
                    q.add(c.right);
            }
            ans.add(t);
        }
        return ans;
    }
}

// 846. Hand of Straights
class Solution {
    public boolean isNStraightHand(int[] hand, int W) {
        TreeMap<Integer, Integer> map = new TreeMap();
        for (int card : hand) {
            if (!map.containsKey(card)) {
                map.put(card, 1);
            } else {
                map.put(card, map.get(card) + 1);
            }
        } // put all the cards and it's frenquency in the TreeMap
          // Java.util.TreeMap uses a red-black tree in the background which makes sure
          // that there are no duplicates;
          // HashMap implements Hashing, while TreeMap implements Red-Black Tree, a Self
          // Balancing Binary Search Tree
          // additionally it also maintains the elements in a sorted order.
        System.out.println(hand.length);
        if (hand.length % W != 0)
            return false;
        // loop through the map for the smallest key
        while (map.size() > 0) {
            int first = map.firstKey();
            for (int k = first; k < first + W; k++) {
                // when still in the loop but consecutive key does not exist, return false;
                if (!map.containsKey(k))
                    return false;
                // check the frequency count;
                int freq = map.get(k);
                // if the frequency == 1, remove this entry form count.
                if (freq == 1)
                    map.remove(k);
                // decrease the frequency count.
                else
                    map.put(k, freq - 1);
            }
        }
        return true;
    }
}

// 01/11/2020

// 235. Lowest Common Ancestor of a Binary Search Tree
/**
 * Definition for a binary tree node. public class TreeNode { int val; TreeNode
 * left; TreeNode right; TreeNode(int x) { val = x; } }
 */
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (p.val < root.val && q.val < root.val) {
            return lowestCommonAncestor(root.left, p, q);
        }

        else if (p.val > root.val && q.val > root.val) {
            return lowestCommonAncestor(root.right, p, q);
        }

        else
            return root;
    }
}

// 236. Lowest Common Ancestor of a Binary Tree
// Assume 2 nodes are both in the tree; this is important! So that you can avoid
// extra work!
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null)
            return null;
        if (root.val == p.val || root.val == q.val)
            return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);

        if (left != null && right != null)
            return root;
        if (left == null && right == null)
            return null;
        return left != null ? left : right;
    }
}

// 108. Convert Sorted Array to Binary Search Tree
class Solution {
    public TreeNode sortedArrayToBST(int[] nums) {
        return sol(nums, 0, nums.length - 1);
    }

    public static TreeNode sol(int[] nums, int start, int end) {

        if (start > end)
            return null;
        // int mid = ((start + end)/2);
        TreeNode root = new TreeNode(nums[((start + end) / 2)]);
        root.left = sol(nums, start, ((start + end) / 2) - 1);
        root.right = sol(nums, ((start + end) / 2) + 1, end);
        return root;

    }
}

// 01/12/2020

// 105. Construct Binary Tree from Preorder and Inorder Traversal
class Solution {

    Map<Integer, Integer> map;

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder == null || inorder == null || preorder.length == 0 || preorder.length != inorder.length)
            return null;

        map = new HashMap<>();

        for (int i = 0; i < inorder.length; i++) {
            map.put(inorder[i], i);
        }

        return build(preorder, inorder, 0, 0, preorder.length - 1);
    }

    public TreeNode build(int[] preorder, int[] inorder, int pre_st, int in_st, int in_end) {
        // in_st = 0, fixed value, when passing as parameter in the recursive function.
        // not apply to right subtree recursion.
        // in this function, we keep checking the indexes in both arrays, without
        // recreating new subarrays, this approach is time efficient.
        if (pre_st > preorder.length || in_st > in_end)
            return null;
        System.out.println(preorder[pre_st]);
        TreeNode root = new TreeNode(preorder[pre_st]);
        // set root_index initial to the start of the inorder array,

        // in preorder, the root always goes first, so the next element in preorder
        // array is the root we need to locate in inorder array.
        // to find the the root in inorder array, we need to check inorder[root_index]
        // == preorder[pre_st] ?
        // so we can locate the current root_index in inorder array.
        // loop is not efficient, we use hashmap.
        // while(root_index<=in_end && inorder[root_index] != preorder[pre_st]) {
        // root_index++;
        // }

        int root_index = map.get(preorder[pre_st]); // pass key get value.

        root.left = build(preorder, inorder, pre_st + 1, in_st, root_index - 1);
        // when we finish the left subtree,
        // for right sub tree, the pre_st = pre_st+(size of left subtree) inthe form of
        // array.
        // the size of left subtree = the size to the left of the root in inorder array.
        root.right = build(preorder, inorder, pre_st + (root_index - in_st) + 1, root_index + 1, in_end);

        return root;
    }
}

// 103. Binary Tree Zigzag Level Order Traversal
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        Stack<TreeNode> s1 = new Stack<>(); //
        Stack<TreeNode> s2 = new Stack<>();

        List<List<Integer>> ans = new ArrayList<>();

        if (root == null)
            return ans;

        s1.add(root); // initial state, add root to s1.

        while (!s1.isEmpty() || !s2.isEmpty()) {
            List<Integer> a1 = new ArrayList<>();
            while (!s1.isEmpty()) {
                TreeNode t = s1.pop();
                a1.add(t.val);
                if (t.left != null)
                    s2.push(t.left);
                if (t.right != null)
                    s2.push(t.right);
            }
            if (!a1.isEmpty())
                ans.add(a1);

            List<Integer> a2 = new ArrayList<>();
            while (!s2.isEmpty()) {
                TreeNode t = s2.pop();
                a2.add(t.val);
                if (t.right != null)
                    s1.push(t.right);
                if (t.left != null)
                    s1.push(t.left);
            }
            if (!a2.isEmpty())
                ans.add(a2);
        }
        return ans;
    }
}

// 01/13/2020
// 116. Populating Next Right Pointers in Each Node
class Solution {
    // Time O(n)，visit each node once
    // Space O(log(n)), perfect tree height at most O(log(n))
    public Node connect(Node root) {
        if (root == null)
            return null;
        if (root.left != null && root.right != null) {
            root.left.next = root.right;
        }
        if (root.next != null && root.right != null) {
            root.right.next = root.next.left;
        }
        connect(root.left);
        connect(root.right);
        return root;
    }
}

// 98. Validate Binary Search Tree
class Solution {
    public boolean isValidBST(TreeNode root) {
        return helper(root, Long.MIN_VALUE, Long.MAX_VALUE);
        // only long type works;
    }

    public boolean helper(TreeNode root, long min, long max) {
        if (root == null)
            return true;
        if (root.val <= min || root.val >= max)
            return false;
        return helper(root.left, min, root.val) && helper(root.right, root.val, max);
    }
}

// 153. Find Minimum in Rotated Sorted Array
// loop through array, find min value;

// 01/14/2020
// 70. Climbing Stairs

// Time O(n); Space O(n); both top-down and botton up
class Solution {
    public int climbStairs(int n) {
        int dp[] = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i < n + 1; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
}

// 322. Coin Change
class Solution {
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        // fill the dp array with amount+1. since # of ways is as much as amount.
        dp[0] = 0;
        // int ans = 0;
        for (int i = 0; i < dp.length; i++) {
            for (int c = 0; c < coins.length; c++) {
                if (i >= coins[c]) {
                    dp[i] = Math.min(dp[i - coins[c]] + 1, dp[i]);
                }
            } // end loop coins array
        } // end finding min for dp[i];
        return dp[amount] > amount ? -1 : dp[amount];
        // dp[amount] > amount, it's going to be -1
        // coin = [3], amount = 2; dp[2] = 3 > amount which is 2.
    }
}

// 01/15/2020
// 1143. Longest Common Subsequence
// Topdown - Time Limit Exceeded
class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
        if (text1.length() == 0 || text2.length() == 0 || text1.isEmpty() || text2.isEmpty())
            return 0;

        // take the substring without the last character.
        String s1 = text1.substring(0, text1.length() - 1);
        System.out.println(s1);
        String s2 = text2.substring(0, text2.length() - 1);
        System.out.println(s2);
        // get the last character for both strings;
        if (text1.charAt(text1.length() - 1) == text2.charAt(text2.length() - 1)) {
            return 1 + longestCommonSubsequence(s1, s2);
        } else {
            return Math.max(longestCommonSubsequence(text1, s2), longestCommonSubsequence(s1, text2));
        }
    }
}

// bottom-up DP Solution
// runtime (O(n*m))
class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
        int n = text1.length();
        int m = text2.length();

        int[][] dp = new int[n + 1][m + 1];
        // length() +1, because we need too consider the empty string.

        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= m; j++) {

                // fill the 2d array with 0 , when either string is empty.
                if (i == 0 || j == 0)
                    dp[i][j] = 0;

                else if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[n][m];
    }
}

// 392. Is Subsequence
// DP solution, bad runtime. O(nm)
class Solution {
    public boolean isSubsequence(String s, String t) {
        if (s.length() == 0)
            return true;
        if (s.length() == 0 && t.length() == 0)
            return true;
        if (s.length() != 0 && t.length() == 0)
            return false;
        if (s.length() > t.length())
            return false;

        int n = s.length();
        int m = t.length();
        // dp table include empty string.
        boolean[][] dp = new boolean[n + 1][m + 1];

        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= m; j++) {
                if (i == 0)
                    dp[i][j] = true;
                else if (j == 0 && i != 0)
                    dp[i][j] = false;
                else if (s.charAt(i - 1) == t.charAt(j - 1)) {
                    dp[i][j] = true && dp[i - 1][j - 1];
                } else {
                    dp[i][j] = dp[i - 1][j] && dp[i][j - 1];
                }
            }
        }
        return dp[n][m];
    }
}

// O(n)
class Solution {
    public boolean isSubsequence(String s, String t) {
        for (int i = 0; i < s.length(); i++) {
            int index = t.indexOf(s.charAt(i));
            // check if character in s is also in t?
            if (index >= 0)
                t = t.substring(index + 1);
            else
                return false;
        }
        return true;
    }
}

// 53. Maximum Subarray
// Time O(n) Space O(1)
class Solution {
    public int maxSubArray(int[] nums) {
        int maxSum = nums[0];
        for (int i = 1; i < nums.length; i++) {
            nums[i] = Math.max(nums[i], nums[i - 1] + nums[i]);
            maxSum = Math.max(maxSum, nums[i]);
        }
        return maxSum;
    }
}

// 01/21/2020
// 62. Unique Paths
// Recursion with memorization
class Solution {
    public int uniquePaths(int m, int n) {
        if (m < 0 || n < 0)
            return 0;
        if (m == 1 && n == 1)
            return 1;
        int[][] path = new int[m + 1][n + 1];
        if (path[m][n] != 0)
            return path[m][n];
        int left = uniquePaths(m - 1, n);
        int up = uniquePaths(m, n - 1);
        path[m][n] = left + up;
        return path[m][n];
    }
}

// DP Solution O(nm)
class Solution {
    public int uniquePaths(int r, int c) { // 3(row) * 2(col) == 2(row) * 3(col)
        if (r == 0 || c == 0)
            return 0;
        int[][] path = new int[r][c];

        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                if (i == 0 || j == 0) {
                    path[i][j] = 1;
                } else {
                    path[i][j] = path[i - 1][j] + path[i][j - 1];
                }
            }
        }
        return path[r - 1][c - 1];
    }
}

// 279. Perfect Squares similar to 322. Coin Change
class Solution {
    public int numSquares(int n) {
        // [1, 2, 3, 4, 5, 6, 7, ...]^2
        // [1, 4, 9, 16, 25, 36, 49, ...]
        double rt = Math.sqrt(n);
        int anchor = (int) Math.floor(rt); // largest sqrt.

        int[] dp = new int[n + 1]; // [13,13,13,13,...]
        Arrays.fill(dp, n + 1);
        dp[0] = 0;

        int[] a = new int[anchor]; // array to store perfect square. [1, 4, 9, 16, 25, 36, 49, ...]
        for (int i = 0; i < anchor; ++i) {
            a[i] = (i + 1) * (i + 1);

        }

        for (int i = 0; i < dp.length; i++) { // i repersents the subcase n.
            for (int j = 0; j < a.length; j++) {
                if (i >= a[j]) { // the number n must > than the perfect square.
                    dp[i] = Math.min(dp[i], dp[i - a[j]] + 1);
                }
            }
        }
        return dp[n];
    }
}

// better
class Solution {
    public int numSquares(int n) {
        int dp[] = new int[n + 1];
        dp[0] = 0;
        for (int i = 1; i <= n; i++) {
            int min = Integer.MAX_VALUE;
            for (int j = 1; j * j <= i; j++) {
                if (i - j * j >= 0)
                    min = Math.min(min, dp[i - j * j]);
            }
            dp[i] = min + 1;
        }
        return dp[n];
    }
}

// 01/22/2020
// 134. Gas Station. Time: O(n) / Space: O(1)
class Solution {
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int total = 0;
        int cur = 0;
        int start = 0;
        for (int i = 0; i < gas.length; i++) {
            cur = cur + gas[i] - cost[i];
            System.out.println("cur: " + cur + "=" + gas[i] + "-" + cost[i]);

            if (cur < 0) {

                System.out.println("start: " + start + " i " + i);
                start = i + 1;
                // must be i+1, not start + 1,
                // if 1st station is (+ gas) but 2nd station become (- gas), start will not
                // update at prev station.
                cur = 0;
            }
            // calculate the total for all the stop, no matter which stop goes first, the
            // total is unchanged.
            total = total + gas[i] - cost[i];
            System.out.println("total: " + total);
        }
        return total >= 0 ? start : -1;
    }
}

// 122. Best Time to Buy and Sell Stock II
// buy and sale can be the same day. need to sale first.
class Solution {
    public int maxProfit(int[] prices) {
        // corner case, null or 0 length
        if (prices.length == 0) {
            return 0;
        }
        int max = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1]) {
                max += prices[i] - prices[i - 1];
            }
        }
        return max;
    }
}

// 01/23/2020
// 714. Best Time to Buy and Sell Stock with Transaction Fee
// Time O(n) / Space O(1)
class Solution {
    public int maxProfit(int[] prices, int fee) {
        int sellforcash = 0, buyorhold = -prices[0];
        // if u have a stoke in your account, all buying, mins from the total from prev
        // day.
        for (int i = 1; i < prices.length; i++) {
            sellforcash = Math.max(sellforcash, buyorhold + prices[i] - fee);
            buyorhold = Math.max(buyorhold, sellforcash - prices[i]);
        }
        return sellforcash;
    }
}

// 309. Best Time to Buy and Sell Stock with Cooldown
class Solution { // hint: A 3 states NFA will help this question.
    public int maxProfit(int[] prices) {

        if (prices.length == 0)
            return 0;

        // sold -> currrent cash amount
        int holdorbuy = -prices[0], sold = 0, rest = 0;
        for (int i = 1; i < prices.length; i++) {
            // use the hold price from previous day[i-1] + prices[i] = profit after sold.
            int prve_sold_price = sold;// 0

            sold = holdorbuy + prices[i];// 1

            holdorbuy = Math.max(holdorbuy, rest - prices[i]);// buy at 1 or by at 2

            rest = Math.max(prve_sold_price, rest);// 0,0
        }
        return Math.max(sold, rest);
    }
}

// 300. Longest Increasing Subsequence
// Time complexity : O(n^2) Two loops of n
// Space complexity : O(n). dp array of size n is used.
class Solution {
    public int lengthOfLIS(int[] nums) {

        if (nums.length == 0)
            return 0;
        if (nums.length == 1)
            return 1;

        int[] dp = new int[nums.length];
        dp[0] = 1;
        int ans = 1;
        for (int i = 1; i < nums.length; i++) {
            int max = 0;
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    // System.out.println("i = "+ i + " " +nums[j] + " < "+ nums[i] + " "+" max: "
                    // +max+" dp["+j+"]:"+dp[j]);
                    max = Math.max(max, dp[j]); // critical point
                    // System.out.println("max : "+max);
                }
            } // end inner for
            dp[i] = max + 1;
            ans = Math.max(ans, dp[i]); // ans is the max in dp array.
        }
        return ans;
    }
}

// 01/27/2020
// 33. Search in Rotated Sorted Array
// Time: O(log(n)); Space O(1)
class Solution {
    public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0)
            return -1;

        int left = 0;
        int right = nums.length - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            // case 1: if n[mid] is the target.
            if (target == nums[mid]) {
                return mid;
            }
            // case 2: [7 0 1 (2) 4 5 6] right half is sorted
            // check sorted half first
            else if (nums[mid] < nums[left]) {
                if (target > nums[mid] && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            // case 3: [4 5 6 (7) 8 1 2] right halfl is sorted
            // check sorted half first
            else {
                if (target >= nums[left] && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
        }
        return -1;
    }
}

// 01/28/2020
// 81. Search in Rotated Sorted Array II
// Time: O(log n); Space O(1)
class Solution {
    public boolean search(int[] nums, int target) {
        // corner case
        if (nums == null || nums.length == 0)
            return false;

        int left = 0;
        int right = nums.length - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            // case 1
            if (nums[mid] == target) {
                return true;
            }
            // added case 4
            // since nums[mid] is not the target, so nums[left] is not target either,
            // we can skip nums[left] by moving the left pointer 1 step to the right.
            else if (nums[mid] == nums[left]) {
                left++;
            }
            // case 2 from #33
            else if (nums[mid] < nums[left]) {
                if (target > nums[mid] && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            // case 3 from #33
            else {
                if (target >= nums[left] && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
        }
        return false;
    }
}

// 153. Find Minimum in Rotated Sorted Array
// Time: O(log n); Space O(1)
class Solution {
    public int findMin(int[] nums) {
        if (nums == null || nums.length == 0)
            return -1;
        if (nums.length == 1)
            return nums[0];
        int left = 0, right = nums.length - 1;

        // csae 1: the array is sorted.
        if (nums[right] > nums[0]) {
            return nums[0];
        }

        while (right >= left) {
            int mid = left + (right - left) / 2;

            // [6 (7) 1] , min is one left to the mid.
            if (nums[mid] > nums[mid + 1]) {
                return nums[mid + 1];
            }

            // [7 (1) 2] , min is at mid.
            if (nums[mid - 1] > nums[mid]) {
                return nums[mid];
            }
            // [4 5 6 (7) 8 1 2]
            if (nums[mid] > nums[left]) {
                left = mid + 1;
            } // [7 0 1 (2) 4 5 6]
            else {
                right = mid - 1;
            }
        }
        return -1;
    }
}

// 34. Find First and Last Position of Element in Sorted Array
// Time: O(2 * log n ) use binary search twice in this algorithm; Space O(1)
class Solution {
    public int[] searchRange(int[] nums, int target) {
        // initialize a array of size 2.
        int[] ans = { -1, -1 };

        ans[0] = helper(nums, target, true);
        ans[1] = helper(nums, target, false);

        return ans;
    }

    public int helper(int[] n, int t, boolean flag) {
        int i = -1;

        int l = 0;
        int r = n.length - 1;

        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (n[mid] == t) {
                if (flag == true) {
                    i = mid;
                    // System.out.println("search left: mid = " + i);
                    r = mid - 1;
                } else {
                    i = mid;
                    // System.out.println("search right: mid" + i);
                    l = mid + 1;
                }
            } else if (t > n[mid]) {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        return i;
    }
}

// 704. Binary Search
class Solution {
    public int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;

        while (left <= right) { // no need to take care of length <2 if we use (left <= right).
            int mid = left + (right - left) / 2;

            if (target == nums[mid]) {
                return mid;
            } else if (target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return -1;
    }
}

// 852. Peak Index in a Mountain Array. easy version of 162
class Solution {
    public int peakIndexInMountainArray(int[] A) {
        int left = 0;
        int right = A.length - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;

            // [1, 2, 3, (4), 1] peek is at left subarray.

            if (A[mid] < A[mid + 1]) {
                left = mid + 1;
            } else {

                // [0, (5), 4, 3, 2, 1, 0] peek is at left subarray.
                right = mid - 1;
            }
        }
        return left;
        // eventually the left pointer will point to the largest element at peek.
    }
}

// 01/30/2020

// 337. House Robber III
// Time O(n); Space:(1)?
/**
 * Definition for a binary tree node. public class TreeNode { int val; TreeNode
 * left; TreeNode right; TreeNode(int x) { val = x; } }
 */
class Solution {
    public int rob(TreeNode root) {
        int result[] = helper(root);
        return Math.max(result[0], result[1]);
    }

    public int[] helper(TreeNode root) {
        if (root == null) {
            return new int[2];
        }
        int[] result = new int[2];
        int[] left = helper(root.left);
        int[] right = helper(root.right);

        result[1] = root.val + left[0] + right[0];
        result[0] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        return result;
    }
}

// 198. House Robber
class Solution {
    public int rob(int[] nums) {

        if (nums == null || nums.length == 0)
            return 0;

        int[] rob = new int[nums.length];
        int[] not = new int[nums.length];

        rob[0] = nums[0];
        not[0] = 0;

        for (int i = 1; i < nums.length; i++) {
            rob[i] = not[i - 1] + nums[i]; // only cares [i-1], so we only need 2 variables, not 2 arrays.
            not[i] = Math.max(rob[i - 1], not[i - 1]);
        }

        return Math.max(rob[nums.length - 1], not[nums.length - 1]);
    }
}

// optimization
class Solution {
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int rob = 0;
        int not = 0;
        for (int n : nums) {
            // house [i-1]
            int prev = Math.max(rob, not);
            // if rob the current house, take not rob the prev house + current house value
            // updata rob value.
            rob = not + n;
            // if not rob the current house, take max from rob or not from previous house
            not = prev;
        }
        return Math.max(rob, not);
    }
}

// 213. House Robber II
class Solution {
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        if (nums.length == 1)
            return nums[0];

        int[] rob1 = new int[nums.length];
        int[] not1 = new int[nums.length];
        int[] rob2 = new int[nums.length];
        int[] not2 = new int[nums.length];

        rob1[0] = nums[0];
        not1[0] = 0;
        rob2[0] = 0;
        not2[0] = 0;

        for (int i = 1; i < nums.length; i++) {
            rob1[i] = not1[i - 1] + nums[i];
            not1[i] = Math.max(rob1[i - 1], not1[i - 1]);

            rob2[i] = not2[i - 1] + nums[i];
            not2[i] = Math.max(rob2[i - 1], not2[i - 1]);

            // System.out.println(rob1[i] + " "+ not1[i] + " " + rob2[i] + " " +not2[i]);
        }
        return Math.max(not1[nums.length - 1], Math.max(rob2[nums.length - 1], not2[nums.length - 1]));
    }
}

// 02/01/2020
// 169. Majority Element Time: O(n) Space: O(n)
class Solution {
    public int majorityElement(int[] nums) {
        if (nums.length == 0 || nums == null)
            return -1;
        if (nums.length == 1)
            return nums[0];
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(nums[i])) {
                map.put(nums[i], map.get(nums[i]) + 1);
                if (map.get(nums[i]) > (nums.length) / 2) {
                    return nums[i];
                }
            } else {
                map.put(nums[i], 1);
            }
        }
        return -1;
    }
}

// 387. First Unique Character in a String
class Solution {
    public int firstUniqChar(String s) {
        Map<Character, Integer> map = new HashMap<>();
        for(int i = 0; i < s.length(); i++) {
            if (map.containsKey(s.charAt(i))){
                map.put(s.charAt(i), map.get(s.charAt(i))+1);
            }
            else {
                map.put(s.charAt(i), 1);
            }
        }
        
        for(int i = 0; i<s.length(); i++){
            if (map.get(s.charAt(i)) == 1)
                return i;
        }
        return -1;
    }
}

// 344. Reverse String  Time: O(n)/ Space: O(1)
class Solution {
    public void reverseString(char[] s) {       
        if (s.length < 2) return;      
        int left = 0;
        int right = s.length-1;
 
        while (left <= right) {
            char t = s[left];
            s[left] = s[right];
            s[right] = t;
            
            left = left + 1;
            right = right - 1;
        }
        return;
    }
}

// 22. Generate Parentheses 
// Space: O(2n) = O(n)  the depth is 2 * n
class Solution {
    public List<String> generateParenthesis(int n) {    
        List<String> ans = new ArrayList<>();
        helper(ans, "", n, n, n);  
        return ans;
    }
    
    public void helper(List<String> ans, String s, int left, int right, int n){
        if (s.length() == n*2) {
            ans.add(s);
            return;
        }
        if(left != 0) { // open parentheses.
            helper(ans, s+"(", left-1, right, n);
        }
        if(left < right){ // the case that we need to close the parentheses.
            helper(ans, s+")", left, right-1, n);
        }
    }   
}

// 2. Add Two Numbers  /  Time: max(m,n) / Space:max(m,n)+1
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        
        if (l1 == null) return l2;
        if (l2 == null) return l1;
        
        ListNode dummy = new ListNode(0); // initialize a dummy node.
        ListNode cur = dummy; // initialize a pointer to the dummy node.
        
        int carry = 0;
        while(l1 != null || l2 != null) {
            int x = (l1 != null) ? l1.val : 0;
            int y = (l2 != null) ? l2.val : 0;
            
            int sum = (x + y + carry) % 10; 
            carry = (x + y + carry) / 10;
            
            ListNode n = new ListNode(sum);
            cur.next = n; // set connection with current node to the new node. dummy -> n
            cur = cur.next; // move the current point to it's next.
            
            if(l1 != null) l1 = l1.next;
            if(l2 != null) l2 = l2.next;
        } 
        // check if the left most digit has carry. append carry to linkedlist.
        if (carry > 0) {
            ListNode c = new ListNode(carry);
            cur.next = c;
            cur = cur.next;
        }
        return dummy.next; // dummy node is 0, skip it when return.
    }
}

// 141. Linked List Cycle  O(n) / O(1)
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) return false;
        ListNode slow = head;
        ListNode fast = head.next; // why fast = head not work?
        
        while (fast != slow) {
            if(fast == null || fast.next == null) return false;
            
            fast = fast.next.next;
            slow = slow.next;
        }
        // when the fast pointer catch the slow pointer, we are out of the loop and return true. 
        return true; 
    }
}

// 242. Valid Anagram
class Solution {
    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) {
        return false;
        }
        Map<Character, Integer> map = new HashMap<>();
        for (char i : s.toCharArray()) {
            if (map.containsKey(i)) {
                map.put(i, map.get(i)+1);
            }else {
                map.put(i, 1);
            }
        }           
        for (char j : t.toCharArray()){
            if (map.containsKey(j)){
                map.put(j, map.get(j)-1);
                if (map.get(j) == 0) { // remove the key when value == 0;
                    map.remove(j);
                }
                
            }else { // if currrent char is not in hashmap, it's not an anagram
                return false;
            }
        }
        // if hashmap is empty, return true, otherwise return false;
        return map.isEmpty() ? true : false;
    }
}

// Time complexity : O(n). Time complexity is O(n) because accessing the counter table is a constant time operation.
// Space complexity : O(1). Although we do use extra space, the space complexity is O(1) because the table's size stays constant no matter how large nn is.
public boolean isAnagram(String s, String t) {
    if (s.length() != t.length()) {
        return false;
    }
    int[] counter = new int[26];
    for (int i = 0; i < s.length(); i++) {
        counter[s.charAt(i) - 'a']++;
        counter[t.charAt(i) - 'a']--;
    }
    for (int count : counter) {
        if (count != 0) {
            return false;
        }
    }
    return true;
}

// 15. 3Sum
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        // [-1, 0, 1, 2, -1, -4]
        Arrays.sort(nums);
        // [-4, -1, -1, 0, 1, 2]
        //       ^            ^
        // corner case: 0,0,0,0

        List<List<Integer>> ans = new ArrayList<>();

        for (int i=0; i < nums.length-2; i++) {
            // nums[i] is the 1st number;
            // check dublicates for the 1st number.
            if (i>0 && nums[i] == nums[i-1]) continue;
            // continue takes you back to the begining of the loop.

            int sumOf2 = 0-nums[i];

            int low = i+1;
            int hi = nums.length-1;
            while (low<hi) {
                if (nums[low] + nums[hi] == sumOf2) {
                    // create inside array and append to the answer array;
                    List<Integer> a = Arrays.asList(nums[i], nums[low], nums[hi]);
                    ans.add(a);
                    // also check dupulicated for the other numbers.
                    while (low<hi && nums[low] == nums[low+1]) low++;
                    while (low<hi && nums[hi] == nums[hi-1]) hi--;
                    // move pointer
                    low++;
                    hi--;
                }else if (nums[low] + nums[hi] < sumOf2){
                    // -1 + 2 = 1 . we need 4.
                    low++;
                }else{
                    hi--;
                }
            }
        }
        return ans;
    }
}

// 66. Plus One
class Solution {
    public int[] plusOne(int[] digits) {  
        int a[] = new int[digits.length+1];       
        int carry = 1, sum = 0;
        for (int i = digits.length-1; i>=0; i--) {          
            sum = digits[i] + carry;            
            a[i+1] = sum % 10;             
            carry = sum / 10;
        }
        
        if (carry != 0) {
            a[0] = 1;
            return a;
        }
        // else return a from index 1 to the end;
        int slice[] = Arrays.copyOfRange(a, 1, a.length);
        return slice;
    }
}
// better
public int[] plusOne(int[] digits) {     
    int n = digits.length;
    for(int i=n-1; i>=0; i--) {
        if(digits[i] < 9) {
            digits[i]++;
            return digits;
        }  
        digits[i] = 0;
    } 
    int[] newNumber = new int [n+1]; // default 0
    newNumber[0] = 1;
    return newNumber;
}

// 207. Course Schedule / Time: O(V+E) vertex + edge why? / Space: O(n)
class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        if (prerequisites == null || prerequisites.length == 0 || prerequisites.length == 1) return true;
        // blueprint:
        // step1: array to keep track of indegree for each course.
        // step2: create a map of each course -> List(courses depend on that)
        // step3: BFS, add to the queue whenever the course's indegree down to 0.
        // step4: check indegree array, if for any course the indegree is not 0, return false.
       
        int indegree[] = new int[numCourses];  
        Map<Integer, List<Integer>> map = new HashMap<>();
        Queue<Integer> queue = new LinkedList<>();
        
        for (int[] i : prerequisites) {
            // step1
            indegree[i[0]]++;
            // step2
            if (!map.containsKey(i[1])) {
                List<Integer> list = new ArrayList<>();        
                // [355, 313] 355 depents on 313
                list.add(i[0]);
                map.put(i[1], list);
            } else {
                // if the course alreadly exist, add it's later course to the list.
                map.get(i[1]).add(i[0]);
            }
        }
        
        // step 3: first add all the Node with indegree 0.
        for (int i = 0; i< indegree.length; i++) {
            if (indegree[i] == 0) queue.offer(i); 
            // what we put in the queue is the index of the value 0, not the actually value.
        }
        
        while(!queue.isEmpty()) {
            int c = queue.poll();
            List<Integer> next = map.get(c); // this will give you array of next classes you can take.
            
            if (next != null) {
                for (int i: next) {                    
                    indegree[i]--;
                    if(indegree[i] == 0) {
                        queue.offer(i); // step3: BFS, add to the queue whenever the course's indegree down to 0.
                    }
                }
            }
        }
        // step 4
        for (int i: indegree){
            if (i != 0) return false;
        }
        return true;
    }
}
// 55. Jump Game   Time: O(n) / Space: O(1)
class Solution {
    public boolean canJump(int[] nums) {
        if (nums == null || nums.length <= 1) return true;     
        int reach=0; 

        // if you are at index i and i > reach, meaning you can not reach this index i in any way, thus we go out of the loop.
        for (int i = 0; i<nums.length && i <= reach; i++) { 
            
            // nums[i] is how many more steps you can jump, + i means you jump from i. 
            reach = Math.max(nums[i] + i, reach);
            // System.out.println("i: " + i + " r: " + reach);
            
            // jump out of the array, of course you reach the end. 
            if (reach >= nums.length-1) return true; 
            
        }
        return false;
    }
}

// 283. Move Zeroes    in-place Time O(n)/ Space O(1)
class Solution {
    public void moveZeroes(int[] nums) {
        int loc = 0;
        for (int i = 0; i<nums.length; i++) {
            if (nums[i] != 0 && i != loc) { // find and nonzero element and fill it to the location in order.           
                nums[loc] = nums[i];
                nums[i] = 0;
                loc++;
            }
            // skip the current element if the location we need to fill is at the curent index.
            else if (nums[i] != 0 && i == loc) { 
                loc++;
            }else continue;
        }
        return;
    }
}

// 448. Find All Numbers Disappeared in an Array
// Time O(n)/  Space O(1) 
class Solution {
    public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> ans = new ArrayList<>();      
        for (int i=0; i<nums.length; i++) {
            //nums[nums[i]-1]
            //nums[i] = 5, nums[5-1]=nums[4] 
            //check if the current value already in the right place, if so, skip 
            while(nums[i] != i+1 && nums[i] != nums[nums[i]-1]){
                int t = nums[i];
                nums[i] = nums[t-1];
                nums[t-1] = t;
            }
        }   
        for (int i = 0; i < nums.length; i++) {           
            // index 0, 1, 2, 3, 4
            // nums  1, 2, 3, 4, 5
            if(nums[i] != i+1) {
                ans.add(i+1);
            }
        }
        return ans;
    }
}

// 234. Palindrome Linked List   Time O(n) / Space O(1)
class Solution {
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) return true;
        
        ListNode fast = head;
        ListNode slow = head;
        
        while(fast != null && fast.next != null ) {
            fast = fast.next.next;
            slow = slow.next;
        }
        
        // reverse the list from slow pointer.
        slow = rev(slow);
        fast = head;
        
        while(slow != null) {
            if (fast.val != slow.val) {
                return false;
            }
            fast = fast.next;
            slow = slow.next;
        }
        return true;
    }
    
    public ListNode rev(ListNode head) {
        // we need a null pointer at the end, do not set prev to head!
        ListNode prev = null;
            
        while (head != null) {
            ListNode n = head.next;
            head.next = prev;
            prev = head; 
            head = n;
        }
        return prev; // not return head
    }
}

// 112. Path Sum
class Solution {     
    public boolean hasPathSum(TreeNode root, int sum) {
        if(root == null) return false;
        // the case you reach the leaf node.
        if(root.left == null && root.right == null && sum-root.val == 0) return true;
        return hasPathSum(root.left, sum-root.val) || hasPathSum(root.right, sum-root.val);
    }
}
// 113. Path Sum II   Time O(n)/ Space ?
class Solution {        
    
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> ans = new ArrayList<>();

        helper(root, sum, new ArrayList<Integer>(), ans);
        return ans;
    }
    
    public void helper(TreeNode root, int sum, List<Integer> cur, List<List<Integer>> ans) {
        if (root == null) return;
        
        cur.add(root.val);
        
        if(root.left == null && root.right == null && root.val == sum) {
            ans.add(cur);
            return;
        }
        
        helper(root.left, sum-root.val, new ArrayList<Integer>(cur), ans); 
        // own copy at each level
        helper(root.right, sum-root.val, new ArrayList<Integer>(cur), ans);
    }
}

// 238. Product of Array Except Self   Time/Space : O(n)
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int len = nums.length;
        int[] L = new int[len];
        int[] R = new int[len];
        
        int[] ans = new int[len];
        
        L[0] = 1;
        for (int i=1; i<len; i++) {
            L[i] = nums[i-1] * L[i-1];
        }
        
        R[len-1] = 1;
        for (int i = len-2; i>=0; i--) {
            R[i] = nums[i+1] * R[i+1];
        }
        
        for (int i=0; i<len; i++) {
            ans[i] = L[i] * R[i];
        }
        
        return ans;
    }
}

// 581. Shortest Unsorted Continuous Subarray  O(n) / O(n)
class Solution {
    public int findUnsortedSubarray(int[] nums) {
        int [] copy = nums.clone();
//         [2, 6, 4, 8, 10, 9, 15]
        Arrays.sort(copy);
//         [2, 4, 6, 8, 9, 10, 15]
        int st = nums.length;
        int end = 0;
        for (int i = 0; i<nums.length; i++) {
            if(copy[i] != nums[i]) {
                st = Math.min(st, i);
                end = Math.max(end, i);
            }
        }
        return end-st > 0 ? end-st+1 : 0;
    }
}

class Solution { // Time O(n) / Space O(1)
    public int findUnsortedSubarray(int[] nums) {
        int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;
        //find min of unsorted part of the array.
        for(int i = 1; i<nums.length; i++) {
            if (nums[i] < nums[i-1]) {
                min = Math.min(min, nums[i]); // 4
            }
        }
        //find max of unsorted part of the array.
        for(int i = nums.length-2; i >= 0; i--) {
            if (nums[i] > nums[i+1]){
                max = Math.max(max, nums[i]); // 10
            }
        }
        int l,r;  
        // min is 4, so when you find a num is > 4, you find the start of the unsorted array.
        for(l=0; l<nums.length; l++) {
            if(min<nums[l]) break;
        }
        //max is 10, when you find 9 < 10, 9's position is the end of the unsorted array.
        for(r=nums.length-1; r>=0; r--) {
            if(max>nums[r]) break;
        }
        return r-l>0 ? r-l+1: 0; 
    }
}

// 48. Rotate Image O(n^2)? / O(1)
class Solution {
    public void rotate(int[][] matrix) {
        int N = matrix.length;
        
        for (int i = 0; i<N; i++) {     
            for (int j = i; j<N; j++) {
                int t = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = t;
            }
        }
        
        for (int i = 0; i<N; i++) {
            for (int j = 0; j<(N/2); j++) {
                int t = matrix[i][j];
                matrix[i][j] = matrix[i][N-1-j];
                matrix[i][N-1-j] = t;
            }
        }
    }
}

// 11. Container With Most Water    Time: O(n) / Space: O(1)
class Solution {  // moving two pointers towards the mid 
    public int maxArea(int[] height) {
        int max = Integer.MIN_VALUE;
        int left = 0;
        int right = height.length-1;
        while(left < right) {
            int area = Math.min(height[left], height[right]) * (right-left); 
            max = Math.max(max, area);
            if (height[left] < height[right]) left++;
            else right--;
        }
        return max;
    }
}


// 739. Daily Temperatures  Time: O(n) / Space O(w)
// The size of the stack is bounded as it represents strictly increasing temperatures.
class Solution {
    public int[] dailyTemperatures(int[] T) {
        int [] ans = new int [T.length];
        Stack<Integer> stack = new Stack<>();
        for (int i = T.length-1; i >= 0; i--) {
            while(!stack.isEmpty() && T[i] >= T[stack.peek()]) {
                // we take out the colder temp from stack, until meet a wammer temp. 
                // The wammer temp we just found is still on the stack.
                stack.pop();
            }
            // if temp on top of the stack is warmer, we have a ans at current position i
            // if no wammer day and the stack is empty, return 0 to the position i.
            ans[i] = stack.isEmpty() ? 0 : stack.peek() - i;         
            stack.push(i); 
            }
            return ans;
        }
}