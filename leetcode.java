
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

// Time: O(n); Space O(n)
// Skewed tree

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {  // BFS
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<>();
        if (root == null) return ans;
        
        Queue<TreeNode> queue = new LinkedList<>();
        // init state, only add root
        queue.offer(root);
        
        while(!queue.isEmpty()){
            // build integer array for each level of the tree.
            List<Integer> level = new ArrayList<>();
            // get current level queue size
            int size = queue.size();      
            // build inner array.
            for(int i = 0; i<size; i++) {
                // get the current node in the queue
                TreeNode n = queue.poll();
                // add it to the inner array
                level.add(n.val);
                if(n.left != null) queue.offer(n.left);
                if(n.right != null) queue.offer(n.right);
            }
            ans.add(level);
        }
        return ans;
    }
}



// /**
//  * Definition for a binary tree node. public class TreeNode { int val; TreeNode
//  * left; TreeNode right; TreeNode(int x) { val = x; } }
//  */
// class Solution {
//     public List<List<Integer>> levelOrder(TreeNode root) {
//         Queue<TreeNode> q = new LinkedList<>();
//         List<List<Integer>> ans = new ArrayList<>();
//         if (root == null) {
//             return ans;
//         }
//         q.add(root);
//         while (!q.isEmpty()) {
//             List<Integer> t = new ArrayList<>();
//             // q.add(root);
//             int size = q.size();
//             for (int i = 0; i < size; i++) {
//                 TreeNode c = q.remove();
//                 t.add(c.val);
//                 if (c.left != null)
//                     q.add(c.left);
//                 if (c.right != null)
//                     q.add(c.right);
//             }
//             ans.add(t);
//         }
//         return ans;
//     }
// }

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
// Time: O(n), go through every node in the tree. Space: O(n) for a linkedlist tree, stack size.
class Solution { // hint: give each node a range
    public boolean isValidBST(TreeNode root) {  
        // pass max and min;
        return dfs(root, null, null);      
    }
    public boolean dfs(TreeNode root, Integer min, Integer max){
        // base case
        if (root == null) return true;
        // return false whenever a node is no valid.
        if ((max != null && root.val >= max) || (min != null && root.val <= min)){
            return false;
        }
        return dfs(root.left, min, root.val) && dfs(root.right, root.val, max);
    }
}

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
    // 1 <= coins.length <= 12 is a constant
    // Runtime: O(n*12), where n is the dp array's length.
    // Space: O(n)
    public int coinChange(int[] coins, int amount) {
        // index represent the amount,
        // value representthe minimal coins conbination, it's the count.
        int dp[] = new int [amount +1];
        
        // amount +1 let us to check if there are no coin cobinations. 
        // the amount is as much as it self.
        Arrays.fill(dp, amount +1);
        
        // zero coin to make amount 0.
        dp[0] = 0;
        
        for (int i = 1; i< amount +1; i++) {
            for (int j = 0; j < coins.length; j++){
                if (i >= coins[j]){
                    // dp[i-coins[j]]+1
                    // go to the proper coin amount in dp array, and +1 means add 1 coin to the result.
                    // always add 1 coin if you can find the subproblem in dp array.
                    dp[i] = Math.min(dp[i], dp[i-coins[j]]+1);
                }
            }
        }       
        return dp[amount] > amount ? -1 : dp[amount];
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
    // Time complexity : O(N) since we go through the string of length N two times.
    // Space complexity : O(1) because English alphabet contains 26 letters.
    public int firstUniqChar(String s) {
        
        Map<Character, Integer> map = new HashMap<>();
            
        for (int i=0; i < s.length(); i++){
            char cur = s.charAt(i);
            if (!map.containsKey(cur)) {
                map.put(cur, 1);
            }else{
                map.put(cur, map.get(cur)+1);
            }
        }    
        // search in hashmap as the string order.
        for (int i=0 ;i<s.length(); i++){
            if(map.get(s.charAt(i)) == 1){
                return i;
            }
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

// 287. Find the Duplicate Number  Time complexity : O(n)/ Space complexity : O(1)
class Solution {
    public int findDuplicate(int[] nums) {
        int slow = nums[0], fast = nums[0];
        do {// do it at least once, cause initially slow == fast;
            // we need to use nums[index] as next step cause we need to 
            //loop through the array more than 1 time. 
            //it's not a linkedlist, so we can not use i++ and j = j+2 to take steps; 
            
            // slow pt takes 1 step;
            slow = nums[slow];
            // fast pt takes 2 steps;
            fast = nums[nums[fast]];
            // System.out.println(" slow " + slow + ", fast " + fast);
        } while(slow != fast) ;
        // when the loop break, the meeting point may not be the duplicated number, 
        // it only shows the cycle has been detected. 

        // 2 pointers take the same steps , break when find duplicated.
        int p1 = nums[0];
        int p2 = slow; // slow pointer is in the cycle.
        
        while (p1!= p2) {
            p1 = nums[p1];
            p2 = nums[p2];
        }
        return p1;       
    }
}


// 437. Path Sum III
// Time Complexity: O(n)
// Space Complexity: O(log n) if balanced tree. O(n) otherwise.  ?
class Solution {
    public int pathSum(TreeNode root, int sum) {
        // the main function returns the count;
        if (root == null) return 0;
        // count form root + count return from left + count return from right.
        // starting point can be any node. so we need to count the case for left and right.
        return dfs(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum);
    }
    // the helper function returns the rest value
    private int dfs(TreeNode root, int sum){
        if (root == null) return 0;
        sum -= root.val;
        // recursive cal
        // if sum == 0, we found a solution, else return 0 solution
        return (sum == 0 ? 1 : 0) + dfs(root.left, sum) + dfs(root.right, sum);
    }
}

// 96. Unique Binary Search Trees
class Solution {  // Time O(n^2) / Space O(n)?
    public int numTrees(int n) {
        // refer back to Catalan Numbers
        // construct a dp array
        int[] dp = new int[n+1];
        //set an initial value for dp[0]
        dp[0] = 1;
        // 1 node can make 1 tree.
        dp[1] = 1;       
        for (int i = 2; i<=n; i++) { // outter loop represents the node at root.
            for (int j =1; j<=i; j++) { 
                // how many ways for certain number of the nodes for left and right sub tree.
                // dp[j-1] --> left
                // dp[i-j] --> right
                dp[i] += dp[j-1] * dp[i-j]; 
            }
        }
        return dp[n];
    }
}

// 78. Subsets   Time: n * 2^n/ Space: O(N)
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>(); // initialize an empty ans array
        if (nums.length == 0 || nums == null) {
            return ans;
        }
        // recusive function parameters: nums array, ans array, current_subset array, index of the current subset
        helper(nums, ans, new ArrayList<>(), 0);
        return ans;
     }    
        private void helper(int[] nums, List<List<Integer>> ans, List<Integer> c_subset, int index) {
            // add the current subset to ans array. 
            // use "new ArrayList<>(c_subset)" to make a deep copy, problem will rise if you use "c_subset"
            // because c_subset is a variable, we will modify c_subset.
            ans.add(new ArrayList<>(c_subset));
            for (int i = index; i < nums.length; i++) {
                c_subset.add(nums[i]);
                // recursive add one more element to the end / DFS
                helper(nums, ans, c_subset, i+1);
                c_subset.remove(c_subset.size()-1);
            }
        }
}

// 90. Subsets II
class Solution {
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>(); // initialize an empty ans array
        if (nums.length == 0 || nums == null) {
            return ans;
        }
        Arrays.sort(nums);
        // recusive function parameters: nums array, ans array, current_subset array, index of the current subset
        helper(nums, ans, new ArrayList<>(), 0);
        return ans;
     }    
        private void helper(int[] nums, List<List<Integer>> ans, List<Integer> c_subset, int index) {
            // add the current subset to ans array. 
            // use "new ArrayList<>(c_subset)" to make a deep copy, problem will rise if you use "c_subset"
            // because c_subset is a variable, we will modify c_subset. (point to the reference)
            ans.add(new ArrayList<>(c_subset));
            for (int i = index; i < nums.length; i++) {
                
                if(i!= index && nums[i] == nums[i-1]) continue; // continue to next iteration of the loop
      
                c_subset.add(nums[i]);

                // recursive add one more element to the end / DFS
                helper(nums, ans, c_subset, i+1);
                c_subset.remove(c_subset.size()-1);
            }
        }
}



// 39. Combination Sum
class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        if(candidates == null || candidates.length == 0) return ans;
        Arrays.sort(candidates);
        // call recursive function to dfs search.
        dfs(candidates, target, 0, new ArrayList<>(), ans);
        return ans;
    }
    
    private void dfs(int[] nums, int target, int index, List<Integer> current, List<List<Integer>> ans) {
        if(target == 0) {
            ans.add(new ArrayList<>(current));
        }
        else if(target>0){
            for (int i = index; i < nums.length; i++){
                if (nums[i] > target) break;
                current.add(nums[i]);
                dfs(nums, target-nums[i], i, current, ans);
                current.remove(current.size()-1);
            }
        }
    }   
}

// 200. Number of Islands
// Time : O(m* n)
// space: O(m * n) recursion stack
class Solution {
    public int numIslands(char[][] grid) {
        if(grid == null || grid.length == 0) return 0;
        int m = grid.length;
        int n = grid[0].length;
        
        int ans = 0;
        
        for (int y=0; y<m; y++){ // check vertical
            for (int x=0; x<n; x++){ // check horizontal          
                if(grid[y][x] == '1') { // if find an island, ans +1
                    ans++;
                    dfs(grid, x, y, n, m);
                }        
            }
        }
        return ans;
    }
    private void dfs(char[][] grid, int x, int y, int n, int m) {
        if (x<0 || y<0 || x>=n || y>=m || grid[y][x] == '0') return; // out of the bound and it's not island 
        grid[y][x] = '0';   // mark every island as seen by turning all 1 to 0;
        dfs(grid, x+1, y, n, m);  // check up, below, left right
        dfs(grid, x-1, y, n, m);
        dfs(grid, x, y+1, n, m);
        dfs(grid, x, y-1, n, m);
    }
}

// 494. Target Sum   Time: O(2^n) / Space: O(n)
class Solution { // dfs no memo, not efficient
    public int findTargetSumWays(int[] nums, int S) {
        if(nums == null || nums.length == 0) return 0;
        return dfs(nums, S, 0);
    }
    
    private int dfs(int[] nums, int S, int index) {
        //return 1(find an ans) or 0(does not add up to S)s
        if(index == nums.length){
            return (S==0) ? 1 : 0;
        }
            
        return dfs(nums, S-nums[index], index+1)
            +dfs(nums, S+nums[index], index+1);
    }
}
// Notice the given condition: The sum of elements in the given array will not exceed 1000.
// means dp[x] < 1000
// Consider dynamic programming 
// come back later




// 114. Flatten Binary Tree to Linked List (in-place)
// Time: O(n)? walk through all n nodes / Space: O(1)
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public void flatten(TreeNode root) {
        if (root == null) return;
        flatten(root.left);
        flatten(root.right);
        
        // nothing to do
        if(root.left == null) return;
        
        // root.left.left is null and return,
        // we want to find the right most leaf of root.left
        TreeNode n = root.left;
        while (n.right != null) {
            n = n.right;
        }
        // append right subtree from the root to the right most leaf's right side.
        n.right = root.right;
        // move the left subtree to right side of the root
        root.right = root.left;
        // left subtree set to nothing.
        root.left = null;
    }
}

// 42. Trapping Rain Water  Time: O(n) / Space: O(1)
class Solution {
    public int trap(int[] height) {
        // collection the result from each index.
        int max = 0;
        // keep update left, right wall max height
        int leftmax = 0; 
        int rightmax = 0;
        // set 2 pointers from left and right.
        int l = 0;
        int r = height.length-1;
        while(l<r){         
            // update max wall height after moving the pointer
            leftmax = Math.max(leftmax, height[l]);
            rightmax = Math.max(rightmax, height[r]);
            
            if(leftmax<rightmax){
                max += Math.min(leftmax,rightmax)-height[l];
                l++;
            }else{
                max += Math.min(leftmax,rightmax)-height[r];
                r--;
            }
        }
        return max;
    }
}


// 75. Sort Colors    Time: O(n) / Space: O(1)
// 2 pass, not efficient 
class Solution {
    public void sortColors(int[] nums) {
        if(nums.length<2 || nums == null) return;
        int c0=0;
        int c1=0;
        int c2=0;
        for (int n: nums){
            if(n == 0) c0++;
            else if(n==1) c1++;
            else c2++;
        }
        System.out.println(c1);
        
        for (int i=0; i< nums.length;i++){
            if (c0 != 0){
                nums[i] = 0;
                c0--;
            } 
            else if (c1 != 0){
                nums[i] = 1;
                c1--;
            } 
            else{
                nums[i] = 2;
                c2--;
            } 
        }
        return;
    }
}

// 1 pass
class Solution {
    public void sortColors(int[] nums) {
        if(nums.length<2 || nums == null) return;
        
        int start = 0;
        int end = nums.length-1;
        int cur = 0;
        
        while(cur <= end){
            // always switch 0 with nums[start]
            if(nums[cur] == 0){
                nums[cur]=nums[start];
                nums[start]=0;
                start++;
                cur++;
            }
            // always switch 2 with nums[end]
            else if(nums[cur] == 2){
                nums[cur]=nums[end];
                nums[end]=2;
                end--;
            }
            else{ // if nums[cur]==1
                cur++;
            }
        }
    }
}

// 64. Minimum Path Sum     Time: O(n)? / Space: O(n)? the input array is 2D
// Time complexity O(N) where N is mxn because we are traversing the whole matrix
// Space Complexity is O(N) where N is mxn because we created another grid to store the values we are computing.

class Solution {
    public int minPathSum(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        // define the state
        int[][] dp = new int [m][n];
        // initialize the upper most row and left most column.
        dp[0][0] = grid[0][0];
        // 1st row
        for(int j = 1; j < n; j++) {
            dp[0][j] = dp[0][j-1] + grid[0][j];
        }
        // 1st column
        for(int i = 1; i < m; i++) {
            dp[i][0] = dp[i-1][0] + grid[i][0];
        }
        // transition function. 
        for (int i = 1; i<m; i++) {
            for (int j = 1; j<n; j++) {
                dp[i][j] = Math.min(dp[i-1][j], dp[i][j-1]) + grid[i][j];
            }
        }
        // return value
        return dp[m-1][n-1];
    }
}


// 560. Subarray Sum Equals K     Time: O(n^2) / Space: O(1)
class Solution {
    public int subarraySum(int[] nums, int k) {
        int count = 0;

        for (int left=0; left<nums.length; left++) {
            int sum = 0;
            for (int right = left; right < nums.length; right++) {
                sum += nums[right];
                if (sum == k) count++;
            }
        }
        
        return count;
    }
}

// 19. Remove Nth Node From End of List  Time: O(L) / Space: O(1) 
// The algorithm makes one traversal of the list of L nodes. Therefore time complexity is O(L) ?
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dum = new ListNode(0, head);
        ListNode slow = dum;
        ListNode fast = dum;
        
        // move the fast pointer to place.
        for (int i=0; i < n; i++){
            fast = fast.next;
        }
        // fast slow move in the same time
        while (fast.next != null){
            fast = fast.next;
            slow = slow.next;
        }
        
        // must point to slow.next.next
        // point to fast will bring error for the corner case when 2 ptrs do not need to move
        slow.next = slow.next.next;
        
        // must return from dum.next, dum node must be excluded.
        return dum.next;
    }
}

// 91. Decode Ways   / Time: O(n) / Space: O(n) 
// 1
class Solution {
    public int numDecodings(String s) {
        int[] dp = new int[s.length()+1];
        dp[0] = 1;
        dp[1] = s.charAt(0) == '0' ? 0 : 1; // if the first char is 0, 0 way to decode, else 1 way to decode.
        
        for (int i = 2; i<= s.length(); i++){
            // int oneDigits = s.charAt(i-1) - '0';
            int oneDigit = Integer.valueOf(s.substring(i-1, i));
            int twoDigits = Integer.valueOf(s.substring(i-2, i));
            
            // is 1 digit case valid?
            if (oneDigits >= 1){
                dp[i] += dp[i-1];
            }
            
            // is the 2 digit case valid?
            if(twoDigits >= 10 && twoDigits <= 26) {
                dp[i] += dp[i-2];
            }

            // no need to takecare of the 0 case, because the array initilized to 0.
        }
        return dp[s.length()];
    }
}
// 2
class Solution {
    public int numDecodings(String s) {
        int[] dp = new int[s.length()+1];
        // corner case
        if (s.charAt(0) == '0') return 0;
        
        dp[0] = 1;
        dp[1] = 1; 
        
        for (int i = 2; i< s.length()+1; i++){
            
            if (s.charAt(i-1) == '0'){ 
                if (s.charAt(i-2) > '2' || s.charAt(i-2) == '0') return 0;
                dp[i] = dp[i-2];
            }else{   
                if (s.charAt(i-2) != '0' && Integer.valueOf(s.substring(i-2, i)) <= 26) {
                    dp[i] = dp[i-1]+dp[i-2];            
                }else{
                    dp[i] = dp[i-1];        
                }
            }
        }
        return dp[s.length()];
    }
}
// 3
class Solution {  Space O(1)
    public int numDecodings(String s) {
        // corner case
        if (s.charAt(0) == '0') return 0;
        
        int prev = 1;
        int cur = 1; 
        
        for (int i = 1; i< s.length(); i++){
            int temp = cur;
            if (s.charAt(i) == '0'){ 
                if (s.charAt(i-1) > '2' || s.charAt(i-1) == '0') return 0;
                // dp[i] = dp[i-2];
                cur = prev;
            }else{   
                if (s.charAt(i-1) != '0' && Integer.valueOf(s.substring(i-1, i+1)) <= 26) {
                    // dp[i] = dp[i-1]+dp[i-2];     
                    cur = cur + prev;
                }
            }
            prev = temp;
        }
        return cur;
    }
}

// 4, as 10/08/2020  O(n)/O(n)
class Solution {
    public int numDecodings(String s) {
        // +1 because we have 0 length add to the front.
  
        int dp[] = new int[s.length()+1];
        
        // when the string length is 0, no way to decode, it count as 1 way.
        dp[0] = 1;
        // when the string length is 1. 
        // check if the number is 0. if 0, then 0 way to decode.
        // 1-9, 1 way to decode.
        dp[1] = s.charAt(0) == '0' ? 0 : 1;
       
        // this is not checking the string contains 0. now lets add the case with 0 present.
        for(int i = 2; i < s.length()+1; i++){
            
            // System.out.println(s.charAt(i-1));
            // invalid case: ...(30), 0 as last digit can not stand alone. return 0;
            // validcase: ...(20) or ...(10), 0 and 2nd last digit must go together. return dp[i-2]
            if (s.charAt(i-1) == '0'){
                // 10, 20
                if(s.charAt(i-2) == '1' || s.charAt(i-2) == '2'){
                    dp[i] = dp[i-2];
                    // System.out.println("dp[i-2]");
                    // System.out.println(dp[i]);
                }
                // 00, 30, 40....90
                else return 0;
            }
            // if current digit is not 0.
            else{ 
                // 2 digit -> 10-26
                if (Integer.valueOf(s.substring(i-2, i)) <= 26 && Integer.valueOf(s.substring(i-2, i)) >= 10){
                    dp[i] = dp[i-1] + dp[i-2];
                    // System.out.println("dp[i-1] + dp[i-2]");
                    // System.out.println(dp[i]);
                }
                
                else {
                    dp[i] = dp[i-1];
                    // System.out.println("dp[i-1]");
                    // System.out.println(dp[i]);
                }
            }
            
            // System.out.println(dp[i]);
        }
        
        return dp[s.length()];
    }
}

// 5. Longest Palindromic Substring

class Solution { // expand around center Time: O(n^2) / Space: O(1) 

    public String longestPalindrome(String s) {
        if (s == null || s.length()< 1) return "";
        int start = 0;
        int end = 0;
        for (int i=0; i<s.length(); i++){
            int len1 = expand(s, i, i);     // odd
            int len2 = expand(s, i, i+1);   // even
            int len = Math.max(len1, len2);
            if (len > end - start) {
                // i is at the center, to find start and end index
                start = i - ((len -1)/2);
                end = i + (len/2);
                System.out.println("start: " + start + " end: " + end);
            }
        }
        return s.substring(start, end+1);
    }
    
    public int expand(String s, int left, int right){
        // invalided bounday  
        if (s == null || left > right) return 0;
        
        // keep expanding until false
        while (left >= 0 && right<s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        System.out.println( " left: " + left + " right: "+ right +  " mines: "+ (right - left - 1));
        return right - left - 1;
    }
}

class Solution { // DP -> O(n^2) / Space: O(n^2) 

    public String longestPalindrome(String s) {
        int n = s.length();
        
        if(s == null || s.length() < 2) {
            return s;
        }
        boolean[][] dp = new boolean[n][n];
        int left = 0;
        int right = 0;
        
        for (int r = 0; r < n; r++){
            for (int l = 0; l < r; l++){
                boolean innerPal = dp[l+1][r-1] || r-l <= 2;
                if (s.charAt(l) == s.charAt(r) && innerPal){
                    dp[l][r] = true;
                    
                    if(r-l > right - left){
                        left = l;
                        right = r;
                    }
                }
            }
        }
        return s.substring(left, right+1);
    }
}


// 647. Palindromic Substrings
class Solution {
    public int countSubstrings(String s) {
        int n = s.length();
        int count = 0;
        boolean [][] dp = new boolean[n][n];
        
        for(int i = 0; i < n;i++) {
            dp[i][i] = true;
            count++;
        }
        
        // outter loop is the right index
        for (int r = 0; r < n; r++){
            // inner loop is the left index
            for (int l = 0; l < r; l++){
                boolean innerPal = dp[l+1][r-1] || r-l <= 2;
                // r-l <= 2 means totsl length is 3, inner is a single char.
                if (s.charAt(r) == s.charAt(l) && innerPal){
                    dp[l][r] = true;
                    count++;                   
                } 
            } 
        }
        return count;
    }
}

class Solution {
    int count;
    public int countSubstrings(String s) {
        count = 0;
        for(int i=0; i<s.length(); i++){
            checkPalindrome(s, i, i);
            checkPalindrome(s, i, i+1);
        }
        return count;
    }
    
    void checkPalindrome(String s, int low, int high){
        if(low > high) return;
        while(low >= 0 && high < s.length() && s.charAt(low) == s.charAt(high)){
            count++;
            low--;
            high++;
        }
    }
}

// 146. LRU Cache    
// HashMap + doubly linkedlist
// less code consider LinkedHashMap 


// Doubly linkedlist: remove a node in Time O(1)
// HashMap: get/put in O(1)
// Space: O(n)

class Node{
    int key, value;
    Node prev, next;
}

class LRUCache {

    // specify all the fields.
    // use final when it's fixed, not going to change.
    final Node head = new Node();
    final Node tail = new Node();
    Map<Integer, Node> map;
    int capacity;
    
    public LRUCache(int capacity) {
        map = new HashMap<>(capacity);
        this.capacity = capacity; 
        // link head and tail when initialize.
        head.next = tail;
        tail.prev = head;
    }
    
    public int get(int key) {
        // get Node form the map, check if the node is null.
        // if the value exist, [remove] the node and [insert] at front, and return the value ; 
        // if not exist, return -1 ;
        
        // Returns the value to which the specified key is mapped, or null if this map contains no mapping for the key.
        Node n = map.get(key);
        int val = -1;
        
        if (n != null){
            remove(n);
            insert(n);
            val = n.value;
        }
        return val;

    }
    
    public void put(int key, int value) {
        // if the key exist, update the value, and [remove] it, [insert] at front. 
        // check the capacity.
        // if exceed the cap. [remove] the last node, and [insert] the new node at front.
        Node n = map.get(key);
        
        if (n == null){
            
            Node i = new Node();
            i.value = value;
            i.key = key;
            
            if (map.size() == this.capacity) {
                // remove last
                map.remove(tail.prev.key); // remove from hashmap !!!!!!!
                remove(tail.prev); // remove from linkedlist
            }       
            // when you get a new key, always put it in the hashmap !!!!!
            map.put(key ,i);
            // insert node i at front of linkedlist;
            insert(i);
        }else{ // key is present, only update the value, no need to check the capacity
            // update the value
            n.value = value;
            // remove it from it's position
            remove(n);
            // insert at front
            insert(n);
        }    
    }
    
    // helper function
    private void insert(Node c){ // add between head and head.next
        c.prev = head;
        c.next = head.next;
        
        head.next = c;
        c.next.prev = c;
        
//         Node head_nx = head.next;
        
//         head.next = c;
//         c.prev = head;
        
//         head_nx.prev = c;
//         c.next = head_nx;
    }
    
    private void remove(Node c){
        // modify c's prev node's and next node's pointer.
        c.prev.next = c.next;
        c.next.prev = c.prev;
        // point c's prev and next to null;
        c.prev = null;
        c.next = null;
        
//         // store 2 nodes before and after c;
//         Node prev_n = c.prev;
//         Node next_n = c.next;
        
//         // link 2 nodes
//         prev_n.next = next_n;
//         next_n.prev = prev_n;      
    }   
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */



class LRUCache {
    // Node class for doubly LinkedList
    class Node{
        int key, value;
        Node prev, next;
        // constructor
        Node(){ 
        }
        Node(int key, int value){
            this.key = key;
            this.value = value;           
        }
    }

    // fields
    final Node head = new Node();
    final Node tail = new Node();
    Map<Integer, Node> map;
    int cap;
    
    public LRUCache(int capacity) {
        map = new HashMap(cap);
        this.cap = capacity;
        // connect the linkedlist
        head.next = tail;
        tail.prev = head;
    }
    
    public int get(int key) {
        int res = -1;
        Node n = map.get(key);
        if (n != null){
            res = n.value;
            // always reorganize the node.
            remove(n);
            add(n);
        }
        return res;
    }
    
    public void put(int key, int value) {
        // check if already exist a key holding the node
        Node e = map.get(key); 
        // if the key already exist, only change the value of the Node.
        if (e != null) {
            e.value = value;
            remove(e);
            add(e);
        }else{
            // check if the cache capacity is full.
            if (map.size() == cap) {
                // remove the node from the map;
                map.remove(tail.prev.key); // refrence to the key.
                // remove the node at the end of the linkedlist
                remove(tail.prev);
            }
            Node n = new Node(key, value);
            // add it to the linkedlist and the hashmap
            add(n);
            map.put(key, n);
        }
        
    }
    
    public void remove(Node n){
        n.prev.next = n.next;
        n.next.prev = n.prev;
        
//         Node nx = n.next;
//         Node p = n.prev;
        
//         nx.prev = p;
//         p.next = nx;
    }
    
    public void add(Node n){
        // always add to the front
        // save head.next before break the link
        Node temp = head.next;
        head.next = temp;
       
        n.prev = head; 
        n.next = temp;
        
        head.next = n;
        temp.prev = n;
    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */

// 127. Word Ladder
class Solution { // basketwang solution
    
    HashMap<String, List<String>> map = new HashMap<>();
    
    public void buildMap(List<String> wordList, String beginWord){
        
        if (!wordList.contains(beginWord)){
            wordList.add(beginWord);
        }
        
        for (String s: wordList){
            List<String> list = new LinkedList<String>();
            // index is the word, add empty list to the map first
            map.put(s, list); 
            // then loop through the wordList find the words only diff in one char
            for (String next: wordList){
                if (diff(s, next) == 1) {  
                    // add to the list if the words only diff in one char
                    map.get(s).add(next);
                }
            }
        } 
    }
    
    public int diff(String s, String t) {
        int count = 0;
        for(int i = 0; i < s.length(); i++){
            if (s.charAt(i) != t.charAt(i)) count++;
        }
        return count;
    }
    
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        if (beginWord.equals(endWord)) return 0;
        buildMap(wordList, beginWord);
        Set<String> set = new HashSet<>();
        Queue<String> queue = new LinkedList<String>();
        
        queue.offer(beginWord);
        set.add(beginWord);
        
        int step = 1;
        while (queue.size() !=0){
            int size = queue.size();
            for(int i = 0; i<size; i++) {
                String cur = queue.poll();
                if (cur.equals(endWord)) return step;
                List<String> next = map.get(cur);
                for (String n: next){
                    if(!set.contains(n)){
                        queue.offer(n);
                        set.add(n);
                    }
                }
            }
            step++;
        }
        return 0;
    }
}


// BFS : using queue; key word: from top to bottom
// DFS : using recursion or stack

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */

//  814. Binary Tree Pruning
class Solution { // DFS:using recursion or stack // BFS: using queue
    public TreeNode pruneTree(TreeNode root) {
        if (root == null) return null;
        dfs(root);
        return root;
    }
    
    public boolean dfs(TreeNode n){
        if (n == null) return false;
        boolean left = dfs(n.left);
        boolean right = dfs(n.right);
        
        // pruning the tree if the return value from recusive call is false 
        if (!left) { 
            n.left = null;
        }
        if (!right) {
            n.right = null;
        }    
        return n.val == 1 || left || right;
    }
}

// Time Complexity: O(N), where N is the number of nodes in the tree. We process each node once.

// Space Complexity: O(H), where H is the height of the tree. This represents the size of the implicit call stack in our recursion.

// 111. Minimum Depth of Binary Tree
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution { // DFS  // Time Complexity: O(N),  // Space Complexity: O(H), call stack in recursion
    public int minDepth(TreeNode root) {
        if (root == null) return 0;
  
        return dfs(root);
        
    }
    public int dfs(TreeNode n){
        if (n == null) return 0;
        
        int left = dfs(n.left);
        int right = dfs(n.right);
        
        if(left == 0) return right + 1;
        if(right == 0) return left + 1;
        
        return Math.min(left, right) + 1;
    }   
}

// try BFS

// 199. Binary Tree Right Side View  
// Time: O(n) for a linkedlist tree, average is O(height); Space O(n) 
class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> ans = new ArrayList<>();
        if (root == null) return ans;
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        
        while(!queue.isEmpty()){
            int size = queue.size();
            for (int i = 0; i < size; i++){
                // take out node form queue
                TreeNode n = queue.poll();
                if (i == size-1) {
                    ans.add(n.val);
                }
                if (n.left != null) {
                    queue.offer(n.left);
                }
                if (n.right != null) {
                    queue.offer(n.right);
                }
            }
        }
        return ans;
    }
}

// 56. Merge Intervals
// Space: O(n) / Time: O(nlogn) for sorting
class Solution {
    public int[][] merge(int[][] intervals) {      
        if (intervals.length <=1) return intervals;
        // sort by start time.
        Arrays.sort(intervals, (a,b)->a[0]-b[0]);    
        List<int[]> res = new ArrayList<>();
                
        res.add(intervals[0]);       
        for (int i = 1; i<intervals.length; i++){
            // merge
            if (intervals[i][0] <= res.get(res.size()-1)[1]){         
                res.get(res.size()-1)[1] = Math.max(res.get(res.size()-1)[1], intervals[i][1]);       
            // add new
            }else{ 
                res.add(intervals[i]);             
            }
        }
        // List -> Array
        return res.toArray(new int[res.size()][]);
    }
}

class Solution {
    public int[][] merge(int[][] intervals) {
        
        List<int[]> ans = new ArrayList<>();
        
        // sort by the starting time. ascending.
        Arrays.sort(intervals, (a, b) -> (a[0] - b[0]));
        
        for(int[] cur: intervals){
            if (ans.isEmpty() || cur[0] > ans.get(ans.size()-1)[1]){
                ans.add(cur);
            }else{ // compare current finish time and modify the finish time of last position of answer array.
                ans.get(ans.size()-1)[1] = Math.max(cur[1], ans.get(ans.size()-1)[1]);
            }
        } 
        return ans.toArray(new int[ans.size()][]);
    }
}

class Solution {
    public int[][] merge(int[][] intervals) {
        if(intervals.length < 2) return intervals;
        Arrays.sort(intervals, (a1, a2) -> Integer.compare(a1[0], a2[0]));
        
        List<int[]> ans = new ArrayList();
        int [] cur = intervals[0];
        ans.add(cur);
        
        for (int[] i : intervals) {
            int st = cur[0];
            int end = cur[1];
            int next_st = i[0];
            int next_end = i[1];
            
            if (end >= next_st) {
                cur[1] = Math.max(end, next_end);
            } else {
                // no need to merge, add to the ans
                cur = i;
                ans.add(cur);
            }
        }
        return ans.toArray(new int[ans.size()][]);
    }
}


// 79. Word Search
// Since the board is n^2, Time: O(n^2); Space O(n^2) -> recusion call stack
// or Time: O(n); Space O(n)-> recursive stack call 
// where n is number of cells on the board, they are the same

class Solution { // not a fast solution
    public boolean exist(char[][] board, String word) {
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                // check if the first letter in the string matches the current letter on the board.
                // then check if the remaining letter matches from the value of dfs method.
                // return true if both conditions satisfied, otherwise, return false.
                if (word.charAt(0) == board[i][j] && dfs(i,j,0, board, word)) return true; 
            }
        }
        return false;
    }  

    public boolean dfs(int i, int j, int index, char[][] board, String word){
        // finish checking every char in the string
        if (index == word.length()) return true;
        // if the go out of the board or the letter does not match the letter at the current index
        if (i<0 || j<0 || i >= board.length || j >= board[i].length || word.charAt(index) != board[i][j]) return false;

        // remember the char in the current cell, then set it to empty for not using again
        // set it back when returen form the recursive call
        char mem = board[i][j];
        board[i][j] = ' ';

        // dfs 4 directions, OR 4 results, only 1 valid will return true
        // index + 1 is for checking the next letter
        boolean exist = dfs(i+1, j, index+1, board, word) ||
                        dfs(i-1, j, index+1, board, word) ||
                        dfs(i, j+1, index+1, board, word) ||
                        dfs(i, j-1, index+1, board, word);

        board[i][j] = mem;

        return exist;
    }   
}


// 202. Happy Number  - Time: O(n), Space: O(n), extra space for hash set? not sure.
class Solution {
    public boolean isHappy(int n) {
        // use set, because if same sum appear(means trap in a cycle),
        // return false if the sum already in the set. this is the time to break the loop.
        Set<Integer> s = new HashSet<>();
        int digit;
        
        while(s.add(n)){
            int sum = 0;
            
            while(n > 0){
                digit = n % 10;
                n = n/10;
                sum += digit*digit;
            }
            n = sum;
            if (sum == 1) return true;
        }
        return false;
    }
}

// 20. Valid Parentheses - Time: O(n) / Space: O(n)
class Solution {
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack();
        for (int i = 0; i< s.length(); i++){
            // when there are element on the stack, check the ASCII # of those parentheses.
            // () differ by 1, {} and [] differ by 2.
            if (!stack.isEmpty() && (s.charAt(i) - stack.peek() == 1 || s.charAt(i) - stack.peek() == 2)){
                stack.pop();
            }else{
                stack.push(s.charAt(i));
            } 
        }
        if(stack.isEmpty()){
            return true;
        }else{
            return false;
        }
    }
}

// 268. Missing Number - Time: O(n), Space: O(1)
class Solution {
    public int missingNumber(int[] nums) {
        // assume theremust be 1 missing number.
        // we are no deal with the case when no number is missing.
        int total=0;
        int n = nums.length;
        
        // from [1, n], we can use the formula to calculate the sum n(n-1)/2.
        // form [0, n], the length of the array + 1, so the sum of no number missing is (n+1)*n/2;
        int sum = (n+1)*n/2;
        
        for(int i = 0; i<nums.length; i++){
            total += nums[i];
        }
        // difference between sum and total is the missing number.
        return sum-total;
    }
}


// 416. Partition Equal Subset Sum

// Time: O(2^n) with out memorization -> n*m?
// space: O(n) where n is the length of the array? I think is n*m the size of the hashmap.


class Solution { // memoization, dp solution will avoid recursion.
    public boolean canPartition(int[] nums) {
      
        int total = 0;
        
        // total is odd, no way to partition.
        for(int n: nums){
            total += n;         
        }
        System.out.println(total);
        if (total % 2 != 0) return false;

        
        return dfs(nums, 0, 0, total/2, new HashMap<String, Boolean>());
    }
    
    
    public boolean dfs(int[] nums, int index, int sum, int half_total, HashMap<String, Boolean> map){
        // record {T or F (value)} at {current index and curren sum (key)}.
        String state = index + " " + sum;
        
        if (map.containsKey(state)) return map.get(state); // return T or F
        
        else {
            if (sum == half_total) return true;      
            if (sum > half_total || index >= nums.length) return false;
        }
        
        // 2 case, include current num or not include
            
        Boolean result = dfs(nums, index+1, sum+nums[index], half_total, map) || dfs(nums, index+1, sum, half_total, map);
        
        map.put (state, result);
        
        return result;
    }
    
}

// 430. Flatten a Multilevel Doubly Linked List 
// Time: O(n)   / Space : O(n)
class Solution {
    public Node flatten(Node head) {
        if (head == null) return null;    
        // set an iterative pointer.
        Node cur = head;     
        // store the leftover Node at current level when branch to the child Node.
        Stack<Node> s = new Stack<>();      
        while(cur != null) {        
            if (cur.child != null){                
                if (cur.next != null){
                    s.push(cur.next);
                } 
                cur.next = cur.child;
                cur.child.prev = cur;
                cur.child = null;
            }
            else {
                if (cur.next == null && !s.isEmpty()){
                    cur.next = s.pop();
                    cur.next.prev = cur;
                }
            }
            cur = cur.next;              
        } 
        // cur is a pointer, move along to the end, so always return head.
        return head;
    }
}

// 162. Find Peak Element
class Solution { // binary search, Time: O(n), Space: O(1)
    public int findPeakElement(int[] nums) {
        int left = 0;
        int right = nums.length-1;
        
        while(left<right){
            
            int mid = (right-left)/2 + left;
                
            if (nums[mid] < nums[mid+1]){
                // the mid is going to land on left side always because integer division.
                // so updata left to mid+1 can prevent us trap in the while loop. 
                left = mid+1; 
            }else{
                right = mid;
            }
        }
        // while loop breaks when left == right, so return left or right is the same.
        return right; 
    }
}

// 24. Swap Nodes in Pairs
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution { // Time: O(n), Space: O(1)
    public ListNode swapPairs(ListNode head) {
        
        // add a dummy node at the front
        ListNode d = new ListNode(0, head);
        
        // cur is a pointer to walk through the list
        ListNode cur = d;
        
        while (cur.next != null && cur.next.next != null){
            
            // p1 is the 1st node in pair, p2 is the 2nd node in pair
            // because we are breaking the link, store the 1st node as p1 (temp node).
            ListNode p1 = cur.next;
            
            // link cur to p2.
            cur.next = cur.next.next; // cur.next.next at this point is p2
            
            // now, link p1 to the node afte p2.
            p1.next = cur.next.next; // cur.next.next at this point is the node after p2.
            
            // link 2nd node's next pointer to p1.
            cur.next.next = p1; //cur.next.next, 2nd next is the "next" pointer.
            
            // skip 2 nodes for next iteration.
            cur = cur.next.next;
        }

        return d.next;
    }
}


// 987. Vertical Order Traversal of a Binary Tree
// assume n nodes. Time: O(n) / Space: O(n) skewed tree
class Solution {
    public List<List<Integer>> verticalTraversal(TreeNode root) {
        
        // the return type
        List<List<Integer>> ans = new ArrayList<>();
        
        // priority queue for sorting when 2 nodes are at the same place.
        // TreeMap is ordered HashMap
        TreeMap<Integer, TreeMap<Integer, PriorityQueue<Integer>>> map = new TreeMap<>();
        
        // call helper function
        dfs(root, 0 , 0, map);
        
        // print every thing in the tree map
        for (TreeMap<Integer, PriorityQueue<Integer>> x : map.values()){

            List<Integer> cur = new ArrayList<>();
            
            for(PriorityQueue<Integer> y: x.values()){
                while(!y.isEmpty()){
                    cur.add(y.poll());
                }     
            } 
            ans.add(cur);   // add inner list to the ans list        
        }
        return ans;
    }
    
    private void dfs(TreeNode root, int x, int y, TreeMap<Integer, TreeMap<Integer, PriorityQueue<Integer>>> map){
        if (root == null) return;
        
        // outter layer, for same x positioon.
        if (!map.containsKey(x)){
            map.put(x, new TreeMap<>());
        }
        
        // inner layer, for same y position
        if (!map.get(x).containsKey(y)){
            map.get(x).put(y, new PriorityQueue<>());
        }
        map.get(x).get(y).offer(root.val);   
        dfs(root.left, x-1, y+1, map);
        dfs(root.right, x+1, y+1, map);        
    }  
}

// 252. Meeting Rooms

class Solution { // Time: O(nlogn)  Space: O(1)
    public boolean canAttendMeetings(int[][] intervals) {      
        if (intervals.length <= 1) return true;        
        // sort by start time.
        // sorting take O(nlogn)
        Arrays.sort(intervals, (a,b) -> Integer.compare(a[0], b[0]));        
        for (int i=1; i<intervals.length; i++){
            // if the end time is less than the start time of the previous meeting. it's overlapped.
            if (intervals[i][0]< intervals[i-1][1]) return false;
        }
        return true;
    }
}

class Solution { // use extra space
    public boolean canAttendMeetings(int[][] intervals) {      
        Set<Integer> set = new HashSet<>();        
        for (int i=0; i<intervals.length; i++){
            int start = intervals[i][0];
            int end = intervals[i][1];
            // add the all the elements form the interval to the set
            // [1,3],[3,6] is a valid solution.
            // Thus, when add intervals, we exclude the last element, [1,3), [3,6)
            for (int j=start; j<end; j++){
                if (set.contains(j)) return false;
                set.add(j);
            }
        }
        return true;
    }
}

// 253. Meeting Rooms II   
// Time:O(nlogn)->sorting / Space: O(n)
// extract-min operation on a heap takes O(logN).
// poll(), offer() all takes O(logn) time. height of the tree. bubble up, bubble done.
class Solution { // use min heap
    public int minMeetingRooms(int[][] intervals) {
        if (intervals.length == 0 || intervals == null) return 0;
        // sort by starting time.
        Arrays.sort(intervals, (a,b)->a[0]-b[0]);
        // min heap
        PriorityQueue<Integer> q = new PriorityQueue<>();
        // offer finish time.
        q.offer(intervals[0][1]);
        
        for (int i=1; i<intervals.length; i++){          
            // current starting time less than the earliest finish time
            if(intervals[i][0] < q.peek()) q.offer(intervals[i][1]);
            else{
                q.poll();
                q.offer(intervals[i][1]);
            }
        }
        return q.size();
    }
}
// Time:O(nlogn)->sorting / Space: O(n)
class Solution {
    public int minMeetingRooms(int[][] intervals) {
        if (intervals == null || intervals.length == 0) return 0; 
        int[] start = new int[intervals.length];
        int[] end = new int[intervals.length];
        
        for (int i=0; i<intervals.length; i++){
            start[i] = intervals[i][0];
            end[i] = intervals[i][1];
        }        
        Arrays.sort(start);
        Arrays.sort(end);       
        
        // point to the index that ends at the current earliest time.
        int earliestFinish = 0;  
        int count = 1; 
        for (int i = 1; i<intervals.length; i++){
            if (start[i] < end[earliestFinish]){
                count++;
            }else{
                earliestFinish++;
            }
        }
        return count;
    }
}

// 1209. Remove All Adjacent Duplicates in String II
// Time: O(n) / Space: O(n) 
class Solution { // best
    public String removeDuplicates(String s, int k) {
        StringBuilder sb = new StringBuilder(s);
        int count[] = new int[sb.length()];
        for (int i=0; i<sb.length(); i++){
            if (i == 0 || sb.charAt(i) != sb.charAt(i-1)){
                count[i] = 1;
            }else{
                count[i] = count[i-1]+1;
                if(count[i] == k){
                    sb.delete(i-k+1, i+1);
                    i= i-k;
                }
            }
        }
        return sb.toString();
    }
}
// 1D candy crash -> remove k or more.
class Solution { // dp approch
    public String removeDuplicates(String s, int k) {
        StringBuilder sb = new StringBuilder(s);
        
        int [] dp = new int[sb.length()];
        
        for(int i = 0; i<sb.length(); i++){
            
            if(i == 0 || sb.charAt(i) != sb.charAt(i-1)){
                dp[i] = 1;
            }else{
                dp[i] = dp[i-1]+1;
                if(dp[i] >= k && (i==sb.length()-1 || sb.charAt(i) != sb.charAt(i+1))) {
                    int count = dp[i];
                    sb.delete(i-count+1, i+1);
                    i = i-count;
                }
            }
        }        
        return sb.toString();
    }
}
class Solution { // stack
    public String removeDuplicates(String s, int k) {
        // 9:13 start
        Stack<Character> s1 = new Stack<>();
        Stack<Integer> s2 = new Stack<>();
        
        s1.push(s.charAt(0));
        s2.push(1);
        for(int i = 1; i<s.length(); i++){
            if (!s1.isEmpty() && s1.peek() == s.charAt(i)){
                int count = s2.pop()+1;
                if(count >= k && (i == s.length()-1 || s.charAt(i) != s.charAt(i+1))){
                    s1.pop();
                }else{
                    s2.push(count);
                }
            }else{
                s1.push(s.charAt(i));
                s2.push(1);
            }
        }       
        StringBuilder sb = new StringBuilder();
        while(!s1.isEmpty()){
            char temp = s1.pop();
            int count = s2.pop();
            for(int i =0; i<count;i++){
                sb.insert(0, temp);
            }
        }      
        return sb.toString();
    }
}



// 797. All Paths From Source to Target  O(2^N * N)
/**?
Typically DFS = O(V + E) => O(n + 2^n) for the graph
There are 2^n paths in a graph, where n is the number of nodes because each node can be in the path or not.

Each path may contain n nodes.
So, map contains n entries where each entry can store 2^n paths of length n.
Space : O(n * 2^n)
**/
class Solution {
    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {      
        List<List<Integer>> ans = new ArrayList<>();       
        // dfs with backtracking. use one path, add or delete a node from that path.
        List<Integer> path = new ArrayList<>();
        path.add(0);    
        dfs(graph, 0, ans, path);       
        return ans;
    }  
    
    private void dfs(int[][] graph, int node, List<List<Integer>> ans, List<Integer> path){
        // base case:
        // if the last node is n-1;
        // what if the last node is not n-1? not matter in this question.
        if (node == graph.length-1) {
            // add current path to the final answer, then backtrack.
            // always make a new list and add to the final answer.
            // the path refer to the pointer memory, if it changes, the final answer will change accordingly.
            ans.add(new ArrayList<>(path));
            return;
        }
        for (int n : graph[node]){
            path.add(n);
            dfs(graph, n, ans, path);
            path.remove(path.size()-1);            
        }
    }
}

// 1029. Two City Scheduling

// sorting always take O(nlogn),
// for a sorted array, overall take O(n)
// Space: O(1)

class Solution {
    public int twoCitySchedCost(int[][] costs) { 
        Arrays.sort(costs, (a,b)-> a[0]-a[1] - (b[0]-b[1]));            
        // costs.forEach(System.out::println);
//         List<Integer> diff = Arrays.stream(costs)
//             .map((cost) -> cost[0]-cost[1])
//             .collect(Collectors.toList());            
//         diff.forEach(System.out::println);      
        int n = costs.length/2;
        int sum = 0;
        for(int i=0; i<n; i++){
            sum += (costs[i][0]+costs[i+n][1]);      
        } 
        return sum;
    }
}

class Solution { // not efficient
    public int twoCitySchedCost(int[][] costs) {
        int sum = 0;   
        Queue<int[]> q = new PriorityQueue<>((a,b) -> a[1]-b[1]);  
        for(int i=0; i<costs.length; i++){
            int diff = costs[i][0] - costs[i][1];
            int[] arr = new int[]{i, diff};
            q.offer(arr);
        }
    
        int count = 0 ; 
        while(!q.isEmpty()){
            int [] cur = q.poll();
            if (count < costs.length/2){
                sum += costs[cur[0]][0];
                System.out.print(costs[cur[0]][0]);
                count++;
            }else{
                sum += costs[cur[0]][1];
                count++;
                System.out.print(costs[cur[0]][1]);
            }
        } 
        return sum;
    }
}

// 1396. Design Underground System -> system design problem
// Time: O(1) for all 3 methods, 
// Space: O(P+S^2), where S is the number of stations on the network, and P is the number of passengers making a journey concurrently during peak time.
// P is the number of passengers making a journey concurrently during peak time.

class UndergroundSystem {
    // Space: O(m+n^2+c),
    // c is number of concurrently checkin's during the peak time.
    // where m is the number of check in we have, it grows as pasengers not leaving the station.
    // n grows as the number of checkout grows. at worst (n choose 2) which is n^2.
    // so we don't know which one is bigger. 
    private Map<Integer, Checkin> checkinMap = new HashMap<>();
    private Map<String, Trip> tripMap = new HashMap<>();
    
    public UndergroundSystem() {
    }
    
    public void checkIn(int id, String stationName, int t) { // O(1)
        checkinMap.put(id, new Checkin(stationName, t));
    }
    
    public void checkOut(int id, String stationName, int t) { // O(1)
        Checkin cur = checkinMap.get(id);
        String startStation = cur.station;
        int startTime = cur.time;
        
        // build trip key.
        String trip = startStation+","+stationName;
        // query the trip info.
        tripMap.putIfAbsent(trip, new Trip());
        tripMap.get(trip).addTrip(t-startTime);
        
        checkinMap.remove(id);
    }
    // Dividing two numbers is also an O(1) operation.
    public double getAverageTime(String startStation, String endStation) {
        String key = startStation + "," + endStation;        
        return tripMap.get(key).getAverage();
    }
       
    class Checkin { // holds the check in information.
        public String station;
        public int time;
        
        public Checkin(String station, int time){ 
            this.station = station;
            this.time = time;
        }
    }
    
    class Trip { // holds average time of completed trip.
        public int count = 0;
        public double total = 0.0;  
        public void addTrip(int duration) {
            count ++;
            total += duration;
        }   
        public double getAverage(){
            return total / count;
        }      
    }
    
}


class UndergroundSystem {
    private Map<String, Pair<Double, Double>> tripMap = new HashMap<>();
    private Map<Integer, Pair<String, Integer>> checkinMap = new HashMap<>();
    
    public UndergroundSystem() {
        
    }
    
    public void checkIn(int id, String stationName, int t) {
        checkinMap.put(id, new Pair(stationName, t));
    }
    
    public void checkOut(int id, String stationName, int t) {
        // lookup the id in checkin map.
        Pair<String, Integer> p = checkinMap.get(id);
        String startStation = p.getKey();
        Integer checkinTime = p.getValue();
        // look up in the journey map.
        
        // make a key for a journey from station A to station B.
        String routeKey = startStation + "," + stationName; //stationName is the end station.
        Pair<Double, Double> routeInfo = tripMap.getOrDefault(routeKey, new Pair<>(0.0, 0.0));
        Double totalTime = routeInfo.getKey();
        Double count = routeInfo.getValue();
        
        // calculate trip time for current id and make changes in tripMap.
        double tripTime = t - checkinTime;
        tripMap.put(routeKey, new Pair<>(totalTime + tripTime, count+1));
        
        checkinMap.remove(id);        
    }
    // Dividing two numbers is also an O(1) operation.
    public double getAverageTime(String startStation, String endStation) {
        String routeKey = startStation + ","+ endStation;
        double avg = tripMap.get(routeKey).getKey() / tripMap.get(routeKey).getValue();
        return avg;
    }
}

/**
 * Your UndergroundSystem object will be instantiated and called as such:
 * UndergroundSystem obj = new UndergroundSystem();
 * obj.checkIn(id,stationName,t);
 * obj.checkOut(id,stationName,t);
 * double param_3 = obj.getAverageTime(startStation,endStation);
 */

 
// 1169. Invalid Transactions
 class Solution { // not efficient
    public List<String> invalidTransactions(String[] transactions) {
        Set<String> set = new HashSet<>();
        
        List<String[]> list = new ArrayList<>();
        
        // parse each string, separeate by ",", store in list.
        for (String tran : transactions){
            
            String[] strs = tran.split(",");
            list.add(strs);
            // for (String i: strs){
            //     System.out.println(i);
            // }            
        }
        // sort Collections by new list by time
        Collections.sort(list, (a,b)->(Integer.parseInt(a[1])-Integer.parseInt(b[1])));
        
        // for (String[] i: list){
        //     for(String j: i){
        //         System.out.println(j);
        //     }   
        // }
        for(int i = 0; i<list.size(); i++){
            String[] cur = list.get(i);
            String s = cur[0] + ","+cur[1] + ","+cur[2] + ","+cur[3];
            // the amount exceeds $1000
            if(Integer.parseInt(cur[2]) > 1000){
                set.add(s);
            }
            // 2 transactions within 60 minutes, same name, different city.
            int j = i-1; // compare with all previuos transactions.
            int time = Integer.parseInt(cur[1]);
            while(j>=0){
                String[] prev = list.get(j);
                String ps = prev[0] + ","+prev[1] + ","+prev[2] + ","+prev[3];
                int prevTime = Integer.parseInt(prev[1]);
                if (time - prevTime > 60) break; // valid for same or different person.
                if (time-prevTime<=60 && prev[0].equals(cur[0]) && !prev[3].equals(cur[3])){
                    // add 2 invalid transactions.
                    if(!set.contains(s)) set.add(s);
                    if(!set.contains(ps)) set.add(ps);
                }
                j--;
            }
        } 
        List<String> res = new ArrayList<>(set);
        return res;
    }
}


// 221. Maximal Square/ Time, Space: O(m*n), as large as size of the matrix.
class Solution {
    public int maximalSquare(char[][] matrix) {
        if (matrix ==null || matrix.length == 0) return 0;
        int [][] dp = new int [matrix.length][matrix[0].length];
        int ans = 0;
        
        for (int i = 0; i<matrix.length; i++){
            for (int j = 0; j<matrix[0].length;j++){
                if (i==0 || j==0) {
                    dp[i][j]= Character.getNumericValue(matrix[i][j]);
                }
                else if (matrix[i][j] == '1'){
                    dp[i][j]= 1 + Math.min(dp[i-1][j-1], Math.min(dp[i][j-1], dp[i-1][j]));
                }else{
                    dp[i][j] = 0;
                }
                // need to update ans for all the cases.
                ans = Math.max(ans, dp[i][j]);
            }
        }
        return ans*ans;
    }
}


// 1629. Slowest Key / Time: O(n) one pass, Space:O(1)
class Solution {
    public char slowestKey(int[] releaseTimes, String keysPressed) {
        char res = keysPressed.charAt(0);
        int max = releaseTimes[0];
        
        for (int i=1; i<releaseTimes.length; i++){
            int cur_duration = releaseTimes[i] - releaseTimes[i-1];
            // System.out.println(cur_duration);
            if (cur_duration > max){
                max = cur_duration;
                res = keysPressed.charAt(i);
                // System.out.println(">");
            }
            if(cur_duration == max){
                char cur = keysPressed.charAt(i);
                // System.out.println("=");
                // System.out.println(cur);
                if (cur-res > 0){
                    res = keysPressed.charAt(i);
                    // System.out.println(res);
                }
            }  
        }
        return res;
    }
}

// Amazon OA2: cutoffRank
public class Solution { // TIME LIMIT EXCEED OR MEMORY EXCEED.

    public static void main(String[] args) {
        int[] nums1 = {100, 50, 50, 25};
        int[] nums2 = {2,2,3,4,5};
        int k1 = 3, k2 = 4;
        System.out.println(cutOffRank(k1,4, nums1));
        // System.out.println("test 2:");
        System.out.println(cutOffRank(k2,5, nums2));
    }
    public int cutOffRank(int cutOffRank, int num, int[] scores) {
        
        TreeMap<Integer, Integer> map = new TreeMap<>((a, b)->b - a);
        int rank = 1;
        int count = 0;

        for(int n : scores) {
        map.put(n, map.getOrDefault(n, 0) + 1);
        }

        for(Map.Entry<Integer, Integer> e : map.entrySet()) { 
            if(rank <= cutOffRank){    
                count += e.getValue();
                rank += e.getValue();
            }else{
                break;
            }
        }
        return count;
    }
}
public class Solution {// Time Complexity: O(n)/ Space Complexity: O(1), score is at most 100, constant
    public int cutOffRank(int cutOffRank, int num, int[] scores) {
        int[] freq = new int[100 + 1];
        for (int score : scores) {
            freq[score]++;
        }  
        int ans = 0;
        int currentRank = 1;

        for (int i = 100; i >= 0 && currentRank <= cutOffRank; i--) {
            if (freq[i] == 0) continue;      
            ans += freq[i];
            currentRank += freq[i];
        }
        return ans;
    }
}

// Amazon OA2: Five-Star Sellers
import java.util.*;

public class Solution {

    public static void main(String[] args) {
        // Test case:
        int[][] reviews1 = {{4,4},{1,2},{3,6}};
        int[][] reviews2 = {{2,4},{1,2}};
        int[][] reviews3 = {{4, 4},{1, 2},{3, 6},{4, 14},{1, 22},{3, 16},{4, 24},{0, 2},{6, 6}};     
        // System.out.println(fiveStar(reviews1, 77)); // Expected: 3
        // System.out.println(fiveStar(reviews2, 51));// Expected: 1
        System.out.println(fiveStarReviews(reviews3, 90)); // Expected: 358
    }

    private static int diff(int[] pair){
        // we choose the rating can increase the most.
        double prev = 1000000 * pair[0]/pair[1]; 
        double cur = 1000000 * (pair[0]+1)/(pair[1]+1);
        return (int)(cur-prev);
    }
        
    public static int fiveStarReviews(int[][] productRatings, int ratingsThreshold) {
        int count = 0;
        double sellerRate = 0.0;
        Queue<int[]> q = new PriorityQueue<>((a,b) -> diff(b) - diff(a));
        for (int[] i : productRatings){
          q.offer(i);
        }
 
        while(sellerRate<ratingsThreshold){
          // calculate current seler rate.
          int[][] arr1 = new int[productRatings.length][productRatings[0].length];
          int[][] arr = q.toArray(arr1);
          sellerRate = 0.0;
          for (int i = 0; i<arr.length;i++){     
            double cur = (arr[i][0] *1.0)/arr[i][1] *100;
            sellerRate += cur;
          } 
          sellerRate /= productRatings.length; 
          // break if reach the threshold. curren queue elements are from the last iteration.
          if (sellerRate>=ratingsThreshold) break;
          // take the product has the most increased rate from queue.
          int[] temp = q.poll();
          // add 1 to both numerator and denominator of the taken product.
          int[] update = {temp[0]+1, temp[1]+1};
          // put it back to the queue. 
          q.offer(update);
          // add 1 to count.
          count++;
        }
        return count;
    }
}


// Amazon OA - Count LRU Cache Misses
import java.util.*;
import java.io.*;
import java.lang.*;

public class Solution {
    public int lruCacheMisses(int num, List<Integer> pages, int maxCacheSize) {
        int miss = 0;
        LinkedList<Integer> list = new LinkedList<>();
        for (Integer i: pages){
            // hit, remove from it's position.
            if (list.contains(i)){
                list.remove(i); // remove the object
            }else{ // whenever the element is not in the cache, miss+1
                miss++;
                if(list.size() == maxCacheSize) list.removeLast();
            }
            // Nomatter hit or miss,move it to the front
            list.addFirst(i);
        }
        return miss;
    }
}
// the reverse will output miss correctly
public class Solution {
    public int lruCacheMisses(int num, List<Integer> pages, int maxCacheSize) {
        int miss = 0;
        List<Integer> list = new LinkedList<>();
        for (Integer i: pages){
            // hit, remove from it's position.
            if (list.contains(i)){
                list.remove(i); // remove the object
            }else{ // whenever the element is not in the cache, miss+1
                miss++;
                if(list.size()== maxCacheSize) list.remove(0); // remove first
            }
            // Nomatter hit or miss, move it, add last
            list.add(i);
        }
        return miss;
    }
}

public class Solution {
    public int lruCacheMisses(int num, List<Integer> pages, int maxCacheSize) {
        int miss = 0;
        List<Integer> list = new LinkedList<>();
        for (int i: pages){
            // hit, remove from it's position.
            if (list.contains(i)){
                list.remove(new Integer(i));
            }else{ // whenever the element is not in the cache, miss+1
                miss++;
                if(list.size()== maxCacheSize) list.remove(0);
            }
            // Nomatter hit or miss, move it to the front
            list.add(i);
        }
        return miss;
    }
}


// 545. Boundary of Binary Tree
// Time complexity : O(n) One complete traversal of the tree is done.
// Space complexity : O(n) The recursive stack can grow upto a depth of n. 

class Solution {   
    List<Integer> ans = new ArrayList<>();   
    public List<Integer> boundaryOfBinaryTree(TreeNode root) { 
        
        if (root == null) return ans;
        // if (root.left != null) 
        ans.add(root.val);
        leftBoundary(root.left);
        leaf(root.left);
        leaf(root.right);
        rightBoundary(root.right);
        
        return ans;
    }
    
    private void leftBoundary(TreeNode root){

        if(root == null) return;        
        if(root.left == null && root.right == null) return;
        
        // add node first.
        ans.add(root.val);
        
        if(root.left == null) leftBoundary(root.right);
        else leftBoundary(root.left);
    }
    
    private void leaf(TreeNode root){
        if(root == null) return;
        if(root.left == null && root.right == null) {
            ans.add(root.val);
            return;
        }
        leaf(root.left);
        leaf(root.right);
    }
    private void rightBoundary(TreeNode root){
        if(root == null) return;
        if(root.left == null && root.right == null) return;
        if(root.right == null) rightBoundary(root.left);
        else rightBoundary(root.right);
        // add node last when return.
        ans.add(root.val);
    }
    
}

// Amazon OA - Multiprocessor System
import java.util.*;
public class Main {
    public static int processor(int[] power, int jobs)
    {
      int time = 0;
      Queue<Integer> q = new PriorityQueue<>((a,b)->b-a);
      for(int i: power){
        System.out.println(i);
        q.offer(i);
      }
      while(jobs > 0){
        int temp = q.poll();
        System.out.println("out: "+temp);
        time++;

        q.offer(temp/2);
        System.out.println("in: "+temp/2);
        jobs -= temp;
      }
      return time;
    }
    public static void main(String[] args) {
        int[] input = {3,1,7,2,4};
        System.out.println(processor(input, 15));
    }
}

// 1071. Greatest Common Divisor of Strings
// Time: O(M+N), M,N are the lengths of str1 and str2
// Space: O(1), max depth of the stack is M+N
class Solution {
    public String gcdOfStrings(String s, String t) {
        // str1 = "ABC", str2 = "ABCABC"
        if (s.length()<t.length()) return gcdOfStrings(t, s);
        
        // str1 = "LEET", str2 = "CODE"
        if (!s.substring(0, t.length()).equals(t)) return "";
        
        // base case
        if (t.isEmpty()) return s;
        
        // s-t
        return gcdOfStrings(s.substring(t.length()), t);
    }
}

// 572. Subtree of Another Tree
// Time complexity : O(m*n). In worst case(skewed tree) traverse function takes O(m*n) time.
// Space complexity : O(n). The depth of the recursion tree can go upto n. n refers to the number of nodes in s.

class Solution {
    public boolean isSubtree(TreeNode s, TreeNode t) {
        // s and t are non-empty.        
        // base case
        if(s==null) return false;
        // if they are the same tree, return true.
        if (sameTree(s,t)){
            return true;
        }else{ // otherwise, find if subtree exist in left subtree or right subtree.
            return isSubtree(s.left, t) || isSubtree(s.right, t);
        }   
    }
    private boolean sameTree(TreeNode s, TreeNode t) {
        // base case
        if(s==null && t==null) return true;
        // base case one of s, t are empty.
        if(s==null || t==null) return false;
        return s.val==t.val && sameTree(s.left, t.left) && sameTree(s.right, t.right);
    }        
}

// 240. Search a 2D Matrix II
// Time: O(row+col)/ Space:O(1)
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int row = matrix.length;
        int col = matrix[0].length;
        
        int i=row-1;
        int j=0;

        //search start from the lower left corner.
        while(i >= 0 && j < col){
            
            if (matrix[i][j] == target) return true;
            if (matrix[i][j] < target) {
                j++;
            }else{
                i--;  
            }
        }
        return false;
    }
}

// Throttling gateway
// Time: O(n) add array to hashmap; loop through every element in the hashmap.
// Space: O(n) create hashmap.
import java.util.*;
public class Main { 
    public static void main(String[] args) {
        // String s = "baccc";
        // System.out.println(label(s, 2));

        int[] t1 = {1,1,1,1,2}; //output=1
        int[] t2 = {1,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,7,11,11,11,11}; // output=7
        int[] t3 = {1,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7}; // output=2

        System.out.println(gateway(t1));
        System.out.println(gateway(t2));
        System.out.println(gateway(t3));
    }
    // Space: O(1), Time: O(n)
    public static int gateway(int[] arriveTime){ 
        int drop = 0;
        for (int i =3; i < arriveTime.length; i++){
          if(arriveTime[i] == arriveTime[i-3]){      
            drop++;
          }else if (i>19 && arriveTime[i] - arriveTime[i-20] < 10) {
            drop++;
          }else if(i>59 && arriveTime[i] - arriveTime[i-60] < 60){
            drop++;
          }else continue;
        }
        return drop;
    } 

    public static int gateway(int[] arriveTime){
        int drop = 0; // re-assigned
        int count10 = 0; // total passengers from time 1. at most 20 in 10 minutes.
        int count60 = 0; // at least 60 in 60 minutes.

        Map<Integer, Integer> map = new HashMap<>();
        
        for (int i : arriveTime){
          map.put(i, map.getOrDefault(i,0)+1);
        }

        int last = arriveTime[arriveTime.length-1];
        for (int i=1; i <= last; i++){ // loop from minute 1 to last minute.
            int cur = (map.get(i)==null)? 0: map.get(i); // get how many people are at given minutes. curre
            if (cur == 0) continue; // no one came in this minute;

            if (i<=10){
              count10 += cur;
            }else{              
              // at time 11, 11-10 = 1
              // at time 21, 21-10 = 11, remove 11
              // at time 100, 100-10=90, remove 90 
              count10 -= map.get(i-10);
              count10 += cur;
            }
            if (i<=60){
              count60 += cur;
            }else{
              count60 -= map.get(i-60);
              count60 += cur;
            }

            System.out.println("Time "+ i + ": cur: "+cur+" ,in 10: "+count10 + " ,in 60: "+ count60);
            
            if (count60 > 60) {
              // count10-60 -> 7,7,7,7 drop last 3 of 7's, not drop all of them.
              // cur = 4 for all other cases
              drop += Math.min(count10-60, cur); // more than 60 people in 60 mins;
              System.out.println("drop 60m: "+drop);
            }else if (count10 > 20){
              drop += Math.min(count10-20, cur);
              System.out.println("drop 10m: "+drop);
            }
            else if (cur > 3) {
              drop += (cur-3);
              System.out.println("drop 1m: "+drop);
            }            
            else{
              continue;
            }
        }
        System.out.println(map);
        return drop;
    }  
}

// Two Sum with unique pairs // or use hashSet for O(n)/ O(n)
// Time: O(nlogn) for sorting
// Space: O(1) 
import java.util.*;
public class Main {
    public static void main(String[] args) {
        int[] nums = {1,1,2,45,46,46}; //output=2
        int target= 47; // 1+46, 2+45
        System.out.println(uniquePair(nums, target));
    }
    public static int uniquePair(int[] nums, int target){
        if (nums == null || nums.length == 0) return 0;
        // need to sort when use pointer.
        Arrays.sort(nums);
        int count = 0;
        int left = 0;
        int right = nums.length-1;
        while(left < right){
          // skip the duplicated number.
          while(nums[left] == nums[left+1]) left++;
          while(nums[right] == nums[right-1]) right--;
          if (nums[left] + nums[right] == target){
              count++;
              left++;
              right--;
          } else if (nums[left] + nums[right] < target){
            left++;
          }else{
            right--;
          }
        }
        return count;
    }  
}

// Unique Device Name // Time: O(n), Space: O(m<n), where m is the count of not unique device as one.
import java.util.*;
public class Main {
    public static void main(String[] args) {       
        String[] names = {"switch", "tv", "switch", "tv","switch", "tv", "tv"};
        System.out.println(deviceName(names));
    }
    public static String deviceName(String[] names){
        // can we modified inplace? in the given array? yes
        // create a hashmap and track the count.
        Map<String, Integer> map = new HashMap<>();

        for (int i = 0; i< names.length; i++){
          
          map.put(names[i], map.getOrDefault(names[i], 0)+1);

          int freq = map.get(names[i]);
          if (freq > 1){
            names[i] += Integer.toString(freq-1);
          }
        }
        return Arrays.toString(names);
    }  
}


// Labeling System
// Time: O(nlogn) for sorting.
// Space: O(n) for new array, actually we can sort des to have O(1) space.

import java.text.CharacterIterator;
import java.util.*;
public class Main {
    public static void main(String[] args) {
        String s = "baccc"; // k=2, output: ccbca
        String s1 = "abbbcccdddddeeeeffff"; // k=2, output: ffeffeededdcddccbbab
        String s2 = "baccczzzzz"; // k=2, output: zzczzczcba
        String s3 = "baccczzzzz"; // k=1, output: zczczczbza
        System.out.println(label(s, 2));
        System.out.println(label(s1, 3));
        System.out.println(label(s2, 2));
        System.out.println(label(s3, 1));
    }
    public static String label(String s, int k){
      char [] arr = s.toCharArray();
      Arrays.sort(arr);
      // char[] d is in decreasing order
      char[] d = new char[arr.length];
      for (int i=0; i<arr.length; i++){
        d[i] = arr[arr.length-1-i];
      }
      int replace = 0;
      for (int i = k; i< d.length; i++){
        // more than k, replece with next large char.
        if(d[i] == d[i-k]){
          replace = i+1;
          while(d[replace] == d[i]){
            replace++;
          }
          System.out.println(d[i]+":"+ d[replace]);
          char temp = d[i];
          d[i] = d[replace];
          d[replace]= temp;
          i = i+k; // less than k, no need to replace, just skip.
        }
      }
      return String.valueOf(d);
    }
}

// Amazon OA2 - Winning Sequence
// Time / Space: O(n)
import java.util.*;
public class Main {
    public static void main(String[] args) {
        // all positive
        // int num=6; int low=3; int up=5; // output = [-1];
        int num=5; int low=3; int up=10; // output = [9,10,9,8,7];
        System.out.println(winningSeq(num, low, up));
    }
    public static List<Integer> winningSeq(int num, int low, int up){
        List<Integer> ans = new ArrayList();
        // n elements can constract an array of size at most 2n-1 for given conditin.
        if(num<3 || (num > 2*(up-low+1)-1)) {
          ans.add(-1);
          return ans;
        }     
        int i = up;
        ans.add(0,i-1); // [9]
        // build the decreasing part.
        while(ans.size() != num && i>=low){
          ans.add(i);
          i--;
        }
        // build the increasing part.
        int j = up-2;
        while(ans.size() != num && j>=low){
          ans.add(0,j);
          j--;
        }
        return ans;
    }
}

// Amazon OA2 - Chemical Delivery System
import java.util.*;
public class Main {
    public static void main(String[] args) {
        // int orders = 4;length of ask 
        int[] ask = {4,6,6,7}; // may not be sorted.
        int type = 3; // [0,1,2]
        // int marks = 9;  // length of markings.
        int[][] marking = {{0,3},{0,5},{0,7},{1,6},{1,8},{1,9},{2,3},{2,5},{2,6}};
        // output = 0 , choose the first flask.
        System.out.println(flask(ask,type, marking));
    }
    public static int flask(int[] ask, int type, int[][] marking){
        int flask = -1;
        // key is flask number, value is total waste.
        TreeMap<Integer, Integer> map = new TreeMap<>();
        // Queue<Integer, Integer> map = new TreeMap<>(); also can use prioority queue.
        int cur_flask = -1; // if cur flask has already calulated, the flag will indicated and we skip.
        for (int i = 0; i<ask.length; i++){
            for (int j = 0; j<marking.length; j++){
              if(marking[j][0] != cur_flask && marking[j][1] >= ask[i]){
                int waste = marking[j][1]-ask[i];
                // System.out.println("ask: "+ask[i]+ ", marking: "+ marking[j][1]+ ", waste: "+ waste+", flask#: "+marking[j][0]);
                map.put(marking[j][0], map.getOrDefault(marking[j][0], 0)+ waste);
                cur_flask = marking[j][0];
              } 
              // (j%type==type-1) only if  # of flask == # of marking for one tpye. otherwise won't work.
              // when the current flask doesn't fit the ask amount, remove the key form the map.
              // invalid flask.
              else if(marking[j][1] < ask[i] && (j==marking.length-1 || marking[j][0]!= marking[j+1][0])){
                // System.out.println("cur_flask: "+cur_flask);
                // System.out.println("marking[j+1][0]: "+marking[j][0]);
                System.out.println("remove: "+marking[j][0]);
                // map.remove(marking[j][0]);
                // instead remove, we add big number to it, make sure it can never be choose.
                map.put(marking[j][0], map.getOrDefault(marking[j][0], 0)+ 10000);
              }
              else continue;
            }
        }
        System.out.println(map);
        flask = map.firstKey();
        return flask;
    }
}
// 88. Merge Sorted Array
// Time: O(m+n) / Space: O(1)
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        if (nums1 == null || m == 0) {
            System.arraycopy(nums2,0,nums1,0,n);
            return;
        }
        if (nums2 == null || n == 0) return;               
        int p2 = n-1;
        int p1 = m-1;
        int p = m+n-1;
        while(p1>=0 && p2>=0){
            if (nums2[p2] > nums1[p1]){
                nums1[p] = nums2[p2];
                p--;
                p2--;
            }else{
                nums1[p] = nums1[p1];
                p--;
                p1--;
            }
        }
        System.out.println(p2);
        System.out.println(p1);
        System.out.println(p);
        System.arraycopy(nums2,0,nums1,0,p2+1);
        return;
    }
}

// 674. Longest Continuous Increasing Subsequence
// Time: O(n)/ Space:O(1)
class Solution {
    public int findLengthOfLCIS(int[] nums) {
        int ans  = 0;
        int start = 0;       
        for(int i = 0; i<nums.length;++i){
            if (i>0 && nums[i] <= nums[i-1]){
                start=i;                 
            }
            ans = Math.max(ans, i-start+1);  
        }
        return ans;
    }
}

// 239. Sliding Window Maximum

class Solution { // dp O(n)
  public int[] maxSlidingWindow(int[] nums, int k) {
    int n = nums.length;
    if (n * k == 0) return new int[0];
    if (k == 1) return nums;

    int [] left = new int[n];
    left[0] = nums[0];
    int [] right = new int[n];
    right[n - 1] = nums[n - 1];
    for (int i = 1; i < n; i++) {
      // from left to right
      if (i % k == 0) left[i] = nums[i];  // block_start
      else left[i] = Math.max(left[i - 1], nums[i]);

      // from right to left
      int j = n - i - 1;
      if ((j + 1) % k == 0) right[j] = nums[j];  // block_end
      else right[j] = Math.max(right[j + 1], nums[j]);
    }

    int [] output = new int[n - k + 1];
    for (int i = 0; i < n - k + 1; i++)
      output[i] = Math.max(left[i + k - 1], right[i]);

    return output;
  }
}


class Solution { // TLE - Time: O(nk)
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        if(n*k==0) return new int[0];      
        int[] ans = new int[n-k+1];
        for(int i=0; i<n-k+1; i++){
            int max = Integer.MIN_VALUE;
            for (int j=i;j<i+k; j++){
                max = Math.max(max,nums[j]);
            }
            ans[i] =max;
        }
        return ans;
    }
}

// Amazon OA2 Divisibility of Strings
import java.util.*;
public class Main {
    public static void main(String[] args) {
       String s = "wqzpuogsqcxpqizenbrhcbijieufuxgqpfijuobkqacjkdnzggijhqurwqyrnejckrnghqsyswhczwdicltjdndaebrtgcysulydcsbupkzogewkqpwtfzvjameircaloaqstsoiepynuvxmmthrsdcvrhdijgvzgmtzeijkzixtfxhrqpllspijwnsitnjazdwqzpuogsqcxpqizenbrhcbijieufuxgqpfijuobkqacjkdnzggijhqurwqyrnejckrnghqsyswhczwdicltjdndaebrtgcysulydcsbupkzogewkqpwtfzvjameircaloaqstsoiepynuvxmmthrsdcvrhdijgvzgmtzeijkzixtfxhrqpllspijwnsitnjazdwqzpuogsqcxpqizenbrhcbijieufuxgqpfijuobkqacjkdnzggijhqurwqyrnejckrnghqsyswhczwdicltjdndaebrtgcysulydcsbupkzogewkqpwtfzvjameircaloaqstsoiepynuvxmmthrsdcvrhdijgvzgmtzeijkzixtfxhrqpllspijwnsitnjazdwqzpuogsqcxpqizenbrhcbijieufuxgqpfijuobkqacjkdnzggijhqurwqyrnejckrnghqsyswhczwdicltjdndaebrtgcysulydcsbupkzogewkqpwtfzvjameircaloaqstsoiepynuvxmmthrsdcvrhdijgvzgmtzeijkzixtfxhrqpllspijwnsitnjazd";

        String t =  "wqzpuogsqcxpqizenbrhcbijieufuxgqpfijuobkqacjkdnzggijhqurwqyrnejckrnghqsyswhczwdicltjdndaebrtgcysulydcsbupkzogewkqpwtfzvjameircaloaqstsoiepynuvxmmthrsdcvrhdijgvzgmtzeijkzixtfxhrqpllspijwnsitnjazdwqzpuogsqcxpqizenbrhcbijieufuxgqpfijuobkqacjkdnzggijhqurwqyrnejckrnghqsyswhczwdicltjdndaebrtgcysulydcsbupkzogewkqpwtfzvjameircaloaqstsoiepynuvxmmthrsdcvrhdijgvzgmtzeijkzixtfxhrqpllspijwnsitnjazd";
       
        // System.out.println(solve("bcdbcdbcd","bacbcd")); // output = -1;
        // System.out.println(solve("bcdbcdbcdbcd","bcdbcd")); // "bcd" output = 3;
        System.out.println(solve(s,t));
    }
    // public static String scd(String s, String t){
    //     if(s.length()<t.length()) return scd(t,s);
    //     if(!s.substring(0, t.length()).equals(t)) return "";
    //     if(t.isEmpty()) return s;
    //     // System.out.println(">>"+s.substring(t.length()));
    //     return scd(s.substring(t.length()), t);
    //     // return s.substring(5);
    // }

    private static int solve(String s1,String s2){

    if(s1.length()%s2.length()!=0) return -1;
    int l2=s2.length();
    for(int i=0;i<s1.length();i++){
        if(s1.charAt(i)!=s2.charAt(i%l2))
            return -1;
    }
    if(s2.length()%2==0){
         if(!s2.substring(0,s2.length()/2).contentEquals(s2.substring(s2.length()/2, s2.length())))
             return s2.length();
    }
    int divisible = 1;
    for(int i=0;i<s2.length()/2;i++){
        if(s2.length()%divisible++ != 0) {
            continue;
        }
        int j=0;
        for(;j<s2.length();j++){
            if(s2.charAt(j)!=s2.charAt(j%(i+1)))
                break;
        }
        if(j==s2.length())
            return i+1;
    }
    return s2.length();
    }
}


// 1328. Break a Palindrome
// Time: O(n)/ Space:O(n)
class Solution {
    public String breakPalindrome(String palindrome) {
        int len = palindrome.length();
        if (len <= 1) return "";
        
        char[] p = palindrome.toCharArray();
        
        for(int i=0; i< len; i++){
            int j = len-1-i; // find the mid point for odd length.
            if(i==j) continue; // chenge mid char doesn't work.
            if(p[i] != 'a'){
                p[i] = 'a';
                return String.valueOf(p);
            }
        }
        // "aaaaa" all a case.
        p[len-1]='b';
        return String.valueOf(p);
    }
}

// 399. Evaluate Division
class Solution {
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        
        double[] ans = new double[queries.size()];
        // when there is no result, return -1.0.
        Arrays.fill(ans, -1.0);
        
        // build a directed graph;
        Map<String, Map<String, Double>> map = new HashMap<>();
        
        for (int i = 0; i < equations.size(); i++){
            String x = equations.get(i).get(0);
            String y = equations.get(i).get(1);
            if (!map.containsKey(x)) map.put(x, new HashMap());
            if (!map.containsKey(y)) map.put(y, new HashMap());
            
            // for the char itslfe.
            map.get(x).put(x, 1.0);
            map.get(y).put(y, 1.0);
            
            map.get(x).put(y, values[i]);
            map.get(y).put(x, 1.0/values[i]);            
        }
        // System.out.println(map);
        
        // dfs search in the map/graph.
        for (int i = 0; i<queries.size(); i++){
            String x = queries.get(i).get(0);
            String y = queries.get(i).get(1);
            if(!map.containsKey(x) || !map.containsKey(y)){
                continue; // default in array is -1.0.
            }else{
                // init: total = 1.0
                double val = dfs(map, x, y, new HashSet<>()); //new HashSet<>() for each element in current map.
                ans[i] = val;
                // System.out.println("ans[i]: "+ans[i]);
            }  
        }
        return ans;
    }
    
    private double dfs(Map<String, Map<String, Double>> map, String x, String y, Set<String> visited){
        // x is the starting point, y is the destination.
        visited.add(x); // mark as seen to avoid the loop.
        
        // base case, not found
        // if (map.get(x) == null || map.get(x).size() == 0) return -1.0;   
        
        // base case, found
        if(map.get(x).containsKey(y)){
            return map.get(x).get(y);
        }
        else{ // check each nodes in the key set.
            for(String i : map.get(x).keySet()){
                // System.out.println("from : " + x);
                if (visited.contains(i)) {
                    // System.out.println("skip i: " + i);
                    continue; // skip the current iteration if already visited.
                }
                // System.out.println("to i: " + i);
                double cur_val = map.get(x).get(i);
                double total = dfs(map, i, y, visited);   
                if (total == -1.0) continue; // reach the end of the function, continue explore the next element.
                return cur_val * total;
            }
        }
        return -1.0; // not found
    } 
}

// 692. Top K Frequent Words
// Time: O(nlogn) / Space: O(n)
class Solution {
    public List<String> topKFrequent(String[] words, int k) {

        Map<String, Integer> map = new HashMap<>();
        
        for (int i=0; i<words.length;i++){
            map.put(words[i], 1+map.getOrDefault(words[i], 0));
        }
        List<String> ans = new ArrayList<>(map.keySet());
        
        // sort by more than one conditions.
        Collections.sort(ans, (a,b)->map.get(a).equals(map.get(b)) ? a.compareTo(b) : map.get(b)-map.get(a));
    
        return ans.subList(0,k);     
    }
}

// 139. Word Break
// Time: O(n^3) There are two nested loops, and substring computation at each iteration. 
// Space: O(n)
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> set = new HashSet<>(wordDict);
        // boolean array default to false;
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;  
        for (int i=0; i<dp.length; i++){
            for (int j=0; j<i; j++){
                // dp[j], the prev word in dict
                // and check the current word.
                if (dp[j] && set.contains(s.substring(j, i))){
                    dp[i]=true;
                    // find one true is ok for current cell.
                    break;
                }
            }
        }
        return dp[s.length()];
    }
}

// 140. Word Break II
class Solution {
    public List<String> wordBreak(String s, List<String> wordDict) {
        Set<String> set = new HashSet<>(wordDict);
        
        HashMap<Integer, List<String>> memo = new HashMap<>();

        return dfs(s, set, 0, memo);
    }
    private List<String> dfs(String s, Set<String> set, int start, HashMap<Integer, List<String>> memo){
        
        if (memo.containsKey(start)){
            return memo.get(start);
        }
        
        List<String> res = new ArrayList<>();
        
        System.out.println("start: "+start);
        
        for(int end = start+1; end<s.length(); end++){
            
            String cur = s.substring(start, end);
            
            if (set.contains(cur)){
                StringBuilder sb = new StringBuilder(cur);
                
                System.out.println("cur: "+cur);
                
                List<String> remains = dfs(s, set, end, memo);
                
                for(String r: remains){
                    sb.append(" ");
                    sb.append(r);
                    res.add(sb.toString());
                    // restore sb after add one result.
                    sb = new StringBuilder(cur);
                }     
            }   
        }
        
        String last = s.substring(start, s.length());
        
        if(set.contains(last)){
            res.add(last);
            // System.out.println("last: " + res);
        }
        System.out.println("res: " + res);
        memo.put(start, res);
        System.out.println(memo);
        return res; 
    }
}



class Solution { // my soln: TLE 
    public List<String> wordBreak(String s, List<String> wordDict) {
        Set<String> set = new HashSet<>(wordDict);
        List<String> ans = new ArrayList<>();
        System.out.println(s.length());
        
        dfs(s, set, ans, 0, 1, new ArrayList<>());
        
        return ans;
    }
    
    private void dfs(String s, Set<String> set, List<String> ans, int start, int end, List<String> temp){

        // System.out.println("start: "+start +  " , end: " + end);
        if (end == s.length()+1 && start+1 == end) {      
            ans.add(String.join(" ", temp));
            // System.out.println(String.join(" ", temp));
            // System.out.println("found one.");
            return;
        }
        if(end == s.length()+1) {
            // System.out.println("the end.");
            return;
        }
        
        String cur = s.substring(start, end);
        
        int prev = start;
        
        if (set.contains(cur)){
            temp.add(cur);
            start = end;

            // System.out.println("cur: " + cur + ", temp: " + temp + ", start: " + start);
            
            dfs(s, set, ans, start, end+1, temp);
            
            // System.out.println("removed: "+temp.get(temp.size()-1));
            temp.remove(temp.size()-1);
            
            dfs(s, set, ans, prev, end+1, temp);
        }else{
            dfs(s, set, ans, start, end+1, temp);
        }  
    }
}

// 445. Add Two Numbers II
// Stack : Time/ Space : O(l1+l2)
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        Stack<Integer> s1 = new Stack<>();  
        Stack<Integer> s2 = new Stack<>();
        
        while(l1 != null){
            s1.push(l1.val);
            l1 = l1.next;
        }
        while(l2 != null){
            s2.push(l2.val);
            l2 = l2.next;
        }        
        ListNode head = null;
        int carry = 0;
        while(!s1.isEmpty() || !s2.isEmpty()){            
            int a = s1.isEmpty() ? 0 : s1.pop();
            int b = s2.isEmpty() ? 0 : s2.pop();
            int sum = a + b + carry;            
            ListNode n = new ListNode(sum % 10);            
            // add to front
            n.next = head;
            // move head pointer to n, the first place.
            head = n;             
            carry = sum/10;
        }        
        if(carry == 1){
            ListNode lead = new ListNode(1);
            lead.next = head;
            head = lead;
        }
        return head;
    }
 }

// Stack : Time : O(l1+l2) / Space : O(max(l1,l2))
 class Solution { // fast
    public ListNode reverseList(ListNode head) {
        ListNode last = null;
        while (head != null) {
            // keep the next node
            ListNode tmp = head.next;
            // reverse the link
            head.next = last;
            // update the last node and the current node
            last = head;
            head = tmp;    
        }    
        return last;
    }
    
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        // reverse lists
        l1 = reverseList(l1);
        l2 = reverseList(l2);
        
        ListNode head = null;
        int carry = 0;
        while (l1 != null || l2 != null) {
            // get the current values 
            int x1 = l1 != null ? l1.val : 0;
            int x2 = l2 != null ? l2.val : 0;
            
            // current sum and carry
            int val = (carry + x1 + x2) % 10;
            carry = (carry + x1 + x2) / 10;
            
            // update the result: add to front
            ListNode curr = new ListNode(val);
            curr.next = head;
            head = curr;
            
            // move to the next elements in the lists
            l1 = l1 != null ? l1.next : null;
            l2 = l2 != null ? l2.next : null;
        }

        if (carry != 0) {
            ListNode curr = new ListNode(carry);
            curr.next = head;
            head = curr;
        }

        return head;
    }
}
// Time/Space: O(l1+l2) 
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode cur = new ListNode();

        Stack<Integer> s1 = new Stack<>();
        Stack<Integer> s2 = new Stack<>();
        
        while(l1 != null){
            s1.push(l1.val);
            l1 = l1.next;
        }
        while(l2 != null){
            s2.push(l2.val);
            l2 = l2.next;
        }
        
        int sum = 0;
        int carry = 0;
        
        while(!s1.isEmpty() || !s2.isEmpty()){
            if(!s1.isEmpty()){
                sum += s1.pop();
            }
            if(!s2.isEmpty()){
                sum += s2.pop();
            }
            sum += carry;
            // for next iteration.
            carry = sum/10; // 0 or 1
            
            // only take one digit to store in the linkedlist.
            sum %= 10; // 0 - 9, if >10, take only last digit
            cur.val = sum;            
            
            // prev node always default to carry. 
            ListNode prev = new ListNode(carry);
            prev.next = cur;
            cur = prev;              
            // restore for next iteration, or sum /= 10
            sum = 0;   
        }
        System.out.println(cur.val);
        return carry==1 ? cur : cur.next;
    }
}
// 3. Longest Substring Without Repeating Characters
// Time: O(n) -> len(s)
// Space: O(26) constant
class Solution {
    public int lengthOfLongestSubstring(String s) {
        // return value is as long as 26.     
        if (s.length() == 0 || s == null) return 0;
        char[] a = s.toCharArray();      
        Map<Character, Integer> map = new HashMap<>();
        map.put(a[0], 0);
        
        int left = 0; // index at the start of new string.        
        int max = 1;
        
        for(int i=1; i< a.length; i++){
            if(map.containsKey(a[i])){
                left = Math.max(map.get(a[i])+1, left);
            }
            max = Math.max(max, i-left+1);             
            map.put(a[i], i); // update anyway or put new key
        }       
        return max;
    }
}

class Solution {
    public int lengthOfLongestSubstring(String s) {
        // return value is as long as 26.     
        if (s.length() == 0 || s == null) return 0;
        int[] dp = new int[s.length()+1];
        dp[0] = 0;
        dp[1] = 1;
        char[] a = s.toCharArray();
        
        Map<Character, Integer> map = new HashMap<>();
        map.put(a[0], 1);
        
        int prev_end = 0; // index at the end of prvious string.        
        for(int i=1; i< a.length;i++){
            if(map.containsKey(a[i])){
                prev_end = Math.max(map.get(a[i]), prev_end);
            }
            map.put(a[i], i+1); // update anyway or put new key
            dp[i+1] = Math.max(dp[i], i-prev_end+1);             
        }       
        return dp[s.length()];
    }
}

// 1472. Design Browser History
// 2 stacks
class BrowserHistory {
    private int cur, end; 
    private String[] history;   
    public BrowserHistory(String homepage) {
        history = new String[5001];
        cur = 0;
        end = 0;
        history[cur] = homepage;
    } 
    public void visit(String url) {
        if (history[cur] != url){
            cur++;
            history[cur] = url;
            end = cur;
        }
    } 
    public String back(int steps) {
        cur = Math.max(0, cur-steps);
        return history[cur];
    }    
    public String forward(int steps) {
        cur = Math.min(end, cur+steps);
        return history[cur];
    }
}

class BrowserHistory {
    Stack<String> visit, backLog;
    String homepage;
    
    public BrowserHistory(String homepage) {
        this.visit = new Stack<>();
        this.backLog = new Stack<>();
        this.homepage = homepage;
        this.visit.push(this.homepage);
    }
    
    public void visit(String url) {
        this.visit.push(url);
        this.backLog.clear();
    }
    
    public String back(int steps) {
        while(this.visit.size()>1 && steps > 0){
            String top = this.visit.pop();
            this.backLog.push(top);
            steps--; 
        }
        return this.visit.peek();
    }
    
    public String forward(int steps) {
        while(!this.backLog.isEmpty() && steps > 0){
            String top_back = this.backLog.pop();
            this.visit.push(top_back);
            steps--;
        }
        return this.visit.peek();
    }
}

// 1244. Design A Leaderboard
// Space: O(n)
class Leaderboard {
    Map<Integer, Integer> map;
    public Leaderboard() {
        map = new HashMap<>();
    }
    public void addScore(int playerId, int score) { //O(1)
        if(map.containsKey(playerId)){
            map.put(playerId, map.get(playerId)+score);
        }else{
            map.put(playerId, score);
        }     
    }
    public int top(int K) { // O(nlogn)
        int sum = 0;
        List<Integer> values = new ArrayList<>(this.map.values());
        Collections.sort(values, Collections.reverseOrder());
        for(int i = 0; i<K; i++){
            sum += values.get(i);
        }
        return sum;
    }
    public void reset(int playerId) { // O(1)
        map.put(playerId, 0);
    }
}

// 314. Binary Tree Vertical Order Traversal
// assume n nodes. Time: O(n) / Space: O(n) skewed tree
class Solution {
    public List<List<Integer>> verticalOrder(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<>();
        if (root == null) return ans;
        TreeMap<Integer, ArrayList> map = new TreeMap<>();
        Queue<Pair<TreeNode, Integer>> q = new LinkedList<>();

        q.offer(new Pair(root, 0));
        
        while(!q.isEmpty()){
            Pair<TreeNode, Integer> cur = q.poll();
            TreeNode n = cur.getKey();
            int col = cur.getValue();       
            map.putIfAbsent(col, new ArrayList());
            map.get(col).add(n.val);
            
            if (n.left != null) q.offer(new Pair(n.left, col-1));
            if (n.right != null) q.offer(new Pair(n.right, col+1));
        }
        for(List<Integer> e: map.values()){
            ans.add(e);
        }       
        return ans;
    }
}

// 390. Elimination Game
// Time : O(logn), binary search
class Solution {
    public int lastRemaining(int n) {
        boolean left = true;
        int remain = n;
        int step = 1;
        int head = 1;
        while(remain>1){
            if (left || remain%2 == 1){
                head += step;
            }
            remain = remain / 2;
            step *= 2;
            left = !left;
        }
        return head;
    }
}
class Solution {
    public int lastRemaining(int n) {
        if(n == 1) return 1;
        if(n % 2 == 1) {
            n--;
        }
        return leftright(n);
    }
    
    private int leftright(int n) {
        if(n == 1) return 1;
        if(n <= 4) return 2;
        return 2 * rightleft(n >> 1);
    }
    
    private int rightleft(int n) {
        if(n <= 2) return 1;
        if(n % 2 == 0) {
            return 2* leftright(n >> 1) - 1;
        } else {
            return 2 * leftright(n >> 1);
        }
    }
} 

// 380. Insert Delete GetRandom O(1)
// Time: O(1)
// Space: O(n)
class RandomizedSet {
    Map<Integer, Integer> map;
    List<Integer> list;
    Random rand = new Random();
    /** Initialize your data structure here. */
    public RandomizedSet() {
        this.map = new HashMap<>();
        this.list = new ArrayList<>();
    }
    
    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
        if(map.containsKey(val)) return false;
        
        map.put(val, list.size());
        list.add(list.size(), val);       
        // System.out.println(map);
        // System.out.println(list);
        return true;
    }
    
    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
        if(!map.containsKey(val)) return false;
            // retrieve index.
            int index = map.get(val);
            // remove in ArrayList. 
            // 1. swap with last element and delete last element.          
            int lastVal = list.get(list.size()-1);  
            list.set(index, lastVal);
            list.remove(list.size()-1);     
            // update the swapped element in the map.
            map.put(lastVal, index);
            // remove in map.
            map.remove(val);     
            // System.out.println("rm: "+map);
            // System.out.println("rm: "+ list);
            return true;  
    } 
    /** Get a random element from the set. */
    public int getRandom() {
        // if (list.size() == 1) return list.get(0);
        // Random rand = new Random(); 
        // int index = rand.nextInt(list.size()-1); 
        // return list.get(index);
        return list.get(rand.nextInt(list.size()));
    }
}

// 394. Decode String
// Time: Time Complexity: \mathcal{O}(\text{maxK} \cdot n)O(maxK⋅n), where \text{maxK}maxK is the maximum value of kk and nn is the length of a given string ss. We traverse a string of size nn and iterate kk times to decode each pattern of form \text{k[string]}k[string]. This gives us worst case time complexity as \mathcal{O}(\text{maxK} \cdot n)O(maxK⋅n).
// Space: O(m+n), where m is the number of letters(a-z) and n is the number of digits(0-9) in string s. In worst case, the maximum size of \text{stringStack}stringStack and \text{countStack}countStack could be m and n respectively.


class Solution {
    public String decodeString(String s) {
        // current string 
        Stack<StringBuilder> s1 = new Stack<>();
        // track count for string after "[".
        Stack<Integer> s2 = new Stack<>();
        
        StringBuilder res = new StringBuilder();
        
        int k = 0;
        
        for (char c: s.toCharArray()){
            if(Character.isDigit(c)){
                k = 10*k + (c -'0'); // 48
            }else if (c == '['){
                s1.push(res);
                res = new StringBuilder();
                s2.push(k);
                k = 0;
            }else if(c == ']'){
                int count = s2.pop();
                StringBuilder prev = s1.pop();                
                for (int i=0; i< count; i++){
                    prev.append(res);
                }
                res=prev;
            }else{ // c == letters
                res.append(c);
            }
        }
        return res.toString();
    }
}


class Solution {
    public String decodeString(String s) {
        if (s== null || s.length() ==0) return s;
        Stack<Integer> s1 = new Stack<>();
        Stack<String> s2 = new Stack<>();
        
        String res = "";
        String num = "";
        
        for(int i = 0; i< s.length(); i++){
            char cur = s.charAt(i);
            if (s1.isEmpty() && Character.isLetter(cur)){
                res += cur;
            }else if(Character.isDigit(cur)){
                num += cur;
                if(!Character.isDigit(s.charAt(i+1))){
                    s1.push(Integer.parseInt(num));
                    num = ""; // reset
                }
            }else if(cur == '[' || Character.isLetter(cur)){
                s2.push(Character.toString(cur));
               
            }else{ // cur == "]", ready to pop
                String temp = "";
                while (!s2.peek().equals("[")){      
                    temp = s2.pop() +temp;
                }
                if(s2.peek().equals("[")){
                    s2.pop();
                    int n = s1.pop();
                    String repeated = temp.repeat(n-1);
                    temp = temp+repeated;
                }
                if(!s1.isEmpty()){
                    s2.push(temp);
                }else{
                    res += temp;   
                }               
            }
        }
        return res;
    }
}

// 92. Reverse Linked List II
class Solution {
    public ListNode reverseBetween(ListNode head, int m, int n) {
        if(head == null) return head;
        ListNode dummy = new ListNode(-1);
        // connect dummy to head.
        dummy.next = head;
        ListNode prev = dummy;
        
        // move prev pointer to m-1 position.
        for(int i=1; i<m; i++){
            prev = prev.next;
        }     
        // find node cur at position m.
        // in this question cur does not change, always at the node originally at m's position, even through the node is moving, the cur pointer always bonded to it.
        // same for prev pointer, always at m-1's position, and that node is not moving at all.
        ListNode cur = prev.next;
        for(int i = m; i<n; i++){
            // next node. prev -> cur -> nx
            ListNode nx = cur.next;
            
            cur.next = nx.next;
            nx.next = prev.next;
            prev.next = nx;
        }
        return dummy.next;
    }
}

// 206. Reverse Linked List
// Time: O(n) / Space: O(1)
class Solution { 
    public ListNode reverseList(ListNode head) {
        
        if(head == null || head.next == null) return head;
        ListNode prev = null;

        while(head != null){
            ListNode nx = head.next;
            head.next = prev;
            prev = head;
            head = nx;
        }
        
        return prev;
    }
}
// recursive
// Time: O(n) / Space: O(1)
public ListNode reverseList(ListNode head) {
    if (head == null || head.next == null) return head;
    ListNode p = reverseList(head.next);
    head.next.next = head;
    head.next = null;
    return p;
}


// 1347. Minimum Number of Steps to Make Two Strings Anagram
// Time: O(n) / Space: O(1) max at 26
class Solution {
    public int minSteps(String s, String t) {
        if(s.length() != t.length()) return -1;
        int[] letters = new int[26];
        for (int i =0; i<s.length(); i++){
            letters[s.charAt(i)-'a']++;
            letters[t.charAt(i)-'a']--;
        }        
        // sum up all positive integers or ABS(all negetive integers) in array.
        int count = Arrays.stream(letters)
            .filter(x -> x>0)
            .sum();      
        return count;
    }
}

// 1188. Design Bounded Blocking Queue
class BoundedBlockingQueue { 
    
    private int size;
    private int capacity;
    private LinkedList<Integer> queue;
    private Object lock;

    public BoundedBlockingQueue(int capacity) {
        this.capacity = capacity;
        this.lock = new Object();
        this.queue = new LinkedList<Integer>();
    }
    
    public void enqueue(int element) throws InterruptedException {
        synchronized(lock) {
            while (this.size == this.capacity) {
                lock.wait();
            }
            queue.add(element);
            this.size++;
            lock.notify();
        }
        
    }
    
    public int dequeue() throws InterruptedException {
        synchronized(lock) {
            while (this.size == 0) {
                lock.wait();
            }
            int element = queue.pollFirst();
            this.size--;
            lock.notify();
            return element;
        }
    }
    
    public int size() {
        return this.size;
    }
}

class BoundedBlockingQueue { // good memory usage.

    private Deque<Integer> q;
    private Semaphore producer;
    private Semaphore consumer;
    
    private ReadWriteLock rwLock;
    
    public BoundedBlockingQueue(int capacity) {
        // The actual queue backing this ADT.
        q = new ArrayDeque<>(capacity);
        
        // Semaphores to represent the spaces in the queue.
        producer = new Semaphore(capacity);
        consumer = new Semaphore(0);
        
        // Granular lock for RW ops on the queue.
        rwLock = new ReentrantReadWriteLock();
    }
    
    public void enqueue(int element) throws InterruptedException {
        // If there is space in the queue, then block it.
        producer.acquire();
        
        // Add the element into the queue.
        rwLock.writeLock().lock();
        q.offer(element);
        rwLock.writeLock().unlock();
        
        // Let a consumer know that an element made it into the queue.
        consumer.release();
    }
    
    public int dequeue() throws InterruptedException {
        // If there is an element in the queue, then reserve it.
        consumer.acquire();
        
        // Remove an element from the queue.
        rwLock.writeLock().lock();
        int result = q.poll();
        rwLock.writeLock().unlock();
        
        // Let a producer know that an empty slot is available for an element.
        producer.release();
        
        return result;
    }
    
    public int size() {
        rwLock.readLock().lock();
        int len = q.size();
        rwLock.readLock().unlock();
        
        return len;
    }
}

// 1429. First Unique Number

class FirstUnique {

  private Queue<Integer> queue = new ArrayDeque<>();
  private Map<Integer, Boolean> isUnique = new HashMap<>();

  public FirstUnique(int[] nums) {
    for (int num : nums) {
      // Notice that we're calling the "add" method of FirstUnique; not of the queue. 
      this.add(num);
    }
  }

  public int showFirstUnique() {
    // We need to start by "cleaning" the queue of any non-uniques at the start.
    // Note that we know that if a value is in the queue, then it is also in
    // isUnique, as the implementation of add() guarantees this.
    while (!queue.isEmpty() && !isUnique.get(queue.peek())) {
      queue.remove();
    }
    // Check if there is still a value left in the queue. There might be no uniques.
    if (!queue.isEmpty()) {
      return queue.peek(); // We don't want to actually *remove* the value.
    }
    return -1;
  }

  public void add(int value) {
    // Case 1: We need to add the number to the queue and mark it as unique. 
    if (!isUnique.containsKey(value)) {
      isUnique.put(value, true);
      queue.add(value);
    // Case 2 and 3: We need to mark the number as no longer unique.
    } else {
      isUnique.put(value, false);
    }
  }
}

class FirstUnique {
    // maintian the order
    private Set<Integer> q = new LinkedHashSet<>();
    private Map<Integer, Boolean> unique = new HashMap<>();
    
    public FirstUnique(int[] nums) { // O(K numbers)
        Arrays.stream(nums)
            .forEach(n -> this.add(n));  // this add is an in clase method.            
    }
    
    public int showFirstUnique() { // O(1)
        if(!q.isEmpty()){
            return q.iterator().next();
        }
        return -1;
    }
    
    public void add(int value) { // O(1)
        // not yet seen
        if (!unique.containsKey(value)){
            unique.put(value, true);
            q.add(value);
        }else{ //already in the queue
            unique.put(value, false);
            q.remove(value);
        }
    }
}


// 333. Largest BST Subtree
class Solution { // fast O(n)
    class Node {
        int size;
        boolean isValid;
        int min;
        int max;

        Node(){
            size = 0;
            isValid = true;
            min = Integer.MAX_VALUE;
            max = Integer.MIN_VALUE;
        }
    }
    public int largestBSTSubtree(TreeNode root) {
        Node ans = postOrder(root);
        return ans.size;
    }
    
    private Node postOrder(TreeNode root){
        Node rNode = new Node();
        if(root != null){
            Node left = postOrder(root.left);
            Node right = postOrder(root.right);
            
            if(left.isValid && right.isValid && root.val > left.max && root.val < right.min){
                rNode.isValid = true;
                rNode.size = left.size + right.size + 1;
                rNode.min = (left.min == Integer.MAX_VALUE)?root.val:left.min;
                rNode.max = (right.max == Integer.MIN_VALUE)?root.val:right.max;
            } else {
                rNode.isValid = false;
                rNode.size = Math.max(left.size, right.size);
            }
        } 
        // System.out.println(rNode.isValid+" | "+rNode.size+" | "+rNode.min+" | "+rNode.max);
        return rNode;
    }
}

// Time: O(n)/ Space: max depth of the tree, worst case can be O(n)
class Solution {
    public int largestBSTSubtree(TreeNode root) {
        if(root == null) {
            return 0;
        }
        
        if(isBST(root, Integer.MAX_VALUE, Integer.MIN_VALUE)){
            return sizeof(root);
        }
        
        return Math.max(largestBSTSubtree(root.left), largestBSTSubtree(root.right));
    }
    
    private int sizeof(TreeNode root){
        if (root == null) return 0;
        return 1 + sizeof(root.left) + sizeof(root.right);
    }
    
    private boolean isBST(TreeNode root, int max, int min){
        if(root == null) return true;
        if(root.val >= max || root.val <= min) return false;
        return isBST(root.left, root.val, min) && isBST(root.right, max, root.val);
    }
}


// 117. Populating Next Right Pointers in Each Node II
// Time/Space: O(n)
class Solution {
    public Node connect(Node root) {
        if (root ==null) return root;
        Queue<Node> q = new LinkedList<>();
        q.offer(root);
        
        while(!q.isEmpty()){
            int size = q.size();

            for (int i=0; i<size; i++){                
                Node cur = q.poll();
                Node nx = q.peek();
                if (i<size-1) {
                    cur.next = nx;
                }
                if(cur.left != null){
                    q.offer(cur.left);
                }
                if(cur.right != null){
                    q.offer(cur.right);
                }
            }
        }
        return root;
    }
}

// 723. Candy Crush
class Solution {
    public int[][] candyCrush(int[][] board) {
        int r = board.length;
        int c = board[0].length;
        Boolean crush = false;
        
        // find if there is anything to crush. mark as -(candy).
        // 1. horizontal
        for(int i=0; i<r; i++){
            for(int j=0; j+2<c; j++){
                int target = Math.abs(board[i][j]);
                if(target!=0 && target == Math.abs(board[i][j+1]) && target == Math.abs(board[i][j+2])){
                    crush = true;
                    board[i][j] = board[i][j+1] = board[i][j+2] = -target;
                }
            }
        }
        // 2. vertical
        for(int i=0; i+2<r; i++){
            for(int j=0; j<c; j++){
                int target = Math.abs(board[i][j]);
                if (target!=0 && target == Math.abs(board[i+1][j]) && target == Math.abs(board[i+2][j])){
                    crush = true;
                    board[i][j] = board[i+1][j] = board [i+2][j] = -target;
                }
            }
        }
        
        // crush the candy and drop the rest candy to fill the space.
        if (crush){
            for(int j=0; j<c; j++){
                int a = r-1; // spot can put candy
                for(int i=r-1; i>=0; i--){
                    if(board[i][j] > 0){
                        board[a][j] = board[i][j];
                        a--;
                    }
                }
                while(a>=0){
                    board[a][j] =0;
                    a--;
                }
            }
        }
        return crush ? candyCrush(board) : board;
    }
}

// 763. Partition Labels
// array is more efficient than hashmap
// Time: O(n)/ Space: O(1): as much as 26
class Solution {
    public List<Integer> partitionLabels(String S) {
        List<Integer> res = new ArrayList<>();
        int[] index = new int[26];        
        for(int i =0; i<S.length(); i++){
            index[S.charAt(i)-'a']=i;
        }
        int start = 0;
        int end = 0;
        for(int i= 0; i<S.length(); i++){
            end = Math.max(end, index[S.charAt(i)-'a']);
            if(i == end){
                res.add(end+1-start);
                start = end+1;
            }
        }
        return res;
    }
}

// 937. Reorder Data in Log Files
// Time: O(nlogn), where n is number of logs in input.
// Arrays.sort(), daul pivit quick sort for primitives.
// Space: O(logn) -> recursive call on quick sort.
class Solution { 
    public String[] reorderLogFiles(String[] logs) {
        // the key is to write a comparator.
        Arrays.sort(logs, (l1, l2) -> {
            // return the index of the 1st occurance.
            int space1 = l1.indexOf(" ");
            String id1 = l1.substring(0, space1);
            String body1 = l1.substring(space1+1);
            
            int space2 = l2.indexOf(" ");
            String id2 = l2.substring(0, space2);
            String body2 = l2.substring(space2+1);
            
            Boolean isDigit1 = Character.isDigit(body1.charAt(0));
            Boolean isDigit2 = Character.isDigit(body2.charAt(0));
            
            // case1: 2 letter logs.
            if(!isDigit1 && !isDigit2){
                int val = body1.compareTo(body2);
                if(val == 0) return id1.compareTo(id2);
                return val;
            }
            // case2: digit vs digit -> leave the original order, return 0.
            // case3: digit vs letter -> letter go first, 1
            // case4: letter vs digit -> -1
            return isDigit1 ? (isDigit2 ? 0 : 1) : -1; 
        });
        return logs;     
    }
}

// 973. K Closest Points to Origin

class Solution { 
    // Time: O(nlogn)? or O(nlogK)
    // Space: O(n)
    public int[][] kClosest(int[][] points, int K) {
        int [][] res = new int [K][2];
       
        PriorityQueue<int[]> q = new PriorityQueue<>((a,b) -> (a[0]*a[0]+a[1]*a[1] - (b[0]*b[0]+b[1]*b[1])));
   
        for(int[] p : points){
            q.offer(p);     
        }
        
        int i = 0;
        while(K!=0){
            res[i]=q.poll();
            i++;
            K--;
        }
        return res;
    }
}

class Solution {
    public int[][] kClosest(int[][] points, int K) {
        // Time: sorting O(nlogn)
        // Space: at most O(n)
        Arrays.sort(points, (a,b) -> (a[0]*a[0]+a[1]*a[1] - (b[0]*b[0]+b[1]*b[1])));
        
        int[][] subarray = IntStream.range(0, K)
                        .mapToObj(i -> points[i])
                        .toArray(int[][]::new);
        // int[][] subarray = Arrays.copyOfRange(points, 0, K);
        return subarray;
    }
}


// 547. Number of Provinces
// Time: O(n^2) / Space: O(n)
class Solution {
    public int findCircleNum(int[][] isConnected) {
        if(isConnected == null || isConnected.length==0) return 0;
        boolean[] visited = new boolean[isConnected.length];
        int res = 0;
        for (int i = 0; i<isConnected.length;i++){          
            if(!visited[i]){
                dfs(isConnected, visited, i);
                res++;
            }
        }
        return res;
    }
    private void dfs(int[][] isConnected, boolean[] visited, int i){
        visited[i] = true;

        for (int j = 0; j<isConnected[0].length; j++){
            if(!visited[j] && isConnected[i][j] == 1){
                dfs(isConnected, visited, j);
            }
        }
    }
}

// 994. Rotting Oranges
// Time: O(m*n) 
// Space: O(m*n) stack on recusive call.
class Solution{
private int row, col;
    public int orangesRotting(int[][] grid) {
        row = grid.length;
        col = grid[0].length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (grid[i][j] == 2) {
                    // dist  = 0;
                    // startingBadOrange = true;
                    dfs(grid, i, j , 0, true);
                }
            }
        }

        int max = 0;
        for (int[] row : grid) {
            for (int c : ints) {
                if (c == 1) {
                    return -1;
                }
                max = Math.min(max, c);
            }
        }
        return -max;
    }
    private void dfs(int[][] grid, int i, int j, int dist, boolean startingBadOrange) {
        if (i >= row || i < 0 || j >= col || j < 0 || grid[i][j] == 0) return;

        if(grid[i][j] < 0 && -grid[i][j] <= dist){
            // at the point that the length of the new route is about to execced the previous min route, no need to continue.
            return;
        }
        if (grid[i][j] == 2 && !startingBadOrange) {
            // wheneverwe meet a bad orange that is not the starting one, return.
            return;
        }
        grid[i][j] = -dist; // inplace mark the current distance.
        dfs(grid, i + 1, j, dist + 1, false);
        dfs(grid, i - 1, j, dist + 1, false);
        dfs(grid, i, j - 1, dist + 1, false);
        dfs(grid, i, j + 1, dist + 1, false);
    }
}

// 819. Most Common Word
class Solution {
    public String mostCommonWord(String paragraph, String[] banned) {
        String[] words = paragraph.replaceAll("[^a-zA-Z0-9]", " ").toLowerCase().split("\\s+");
        
        Set<String> set = new HashSet<String>(Arrays.asList(banned));
        
        Map<String, Integer> map = new HashMap<>();
        
        for(String s: words){
            if (!set.contains(s)){ // filter out the banned word
                map.put(s, map.getOrDefault(s, 0)+1);
            } 
        }

        return Collections.max(map.entrySet(), Map.Entry.comparingByValue()).getKey();
    }
}

// 682. Baseball Game
// Time/ Space : O(n)

class Solution { // array is fast than stack
    public int calPoints(String[] ops) {
        
        int[] arr = new int[ops.length];
        int sum = 0;
        int p = -1;
        
        for (String s : ops) {
            char c = s.charAt(0);
            
            if (c == '+') {
            
                int plus = arr[p-1]+arr[p];
                p++;
                arr[p] = plus;
                sum+=plus;
                
            } else if (c == 'D') {
               int db = arr[p] *2;
                p++;
                arr[p] = db;
                sum+=db;
            } else if (c == 'C') {
               int minus = arr[p];
                p--;
            
                sum -= minus;
                
            } else {
                int val = Integer.parseInt(s);
                p++;
                arr[p] = val;
                sum += val;
            }
        }
        
        // for(int i : arr){
        //     System.out.println(i);
        // }
        
        return sum;
    }
}
class Solution {
    public int calPoints(String[] ops) {
        int res = 0;
        Stack<Integer> stack = new Stack<>();       
        for(String s : ops){
            if (s.equals("C")){
                stack.pop();
            }
            else if (s.equals("D")){
                int val = stack.peek() * 2;
                stack.push(val);
            }
            else if (s.equals("+")){
                // a+b=c
                int b = stack.pop();
                int c = b + stack.peek();
                stack.push(b);
                stack.push(c);
            }else{ // digit
                stack.push(Integer.parseInt(s));
            }
        }
        // don't have to pop from stack, just loop through it.
        for(int i : stack) res+=i;
        return res;
    }
}

// 138. Copy List with Random Pointer
class Solution { // Time: O(n) / Space: O(1)
    public Node copyRandomList(Node head) {
        if (head == null) return null;
        Node cur = head;

        // pass 1: create copy and insert it into the given list.        
        while(cur != null){
            Node copy = new Node(cur.val);
            copy.next = cur.next;
            cur.next = copy;
            cur = copy.next;
        }
        
        // pass 2: wire random for the copys.
        cur = head;
        while(cur != null){
            // get cur node's random, if it's not null, then it's next is the copy's random.
            cur.next.random = (cur.random == null) ? null : cur.random.next;        
            cur = cur.next.next;// skip 2 places.
        }
        
        // pass 3: reconnect the node. 1 list -> 2 list.
        cur = head;
        Node res = head.next;
        while(cur != null){
            Node cp = cur.next;
            Node nx = cur.next.next; // the next node of original list.
            if(nx == null){
                cur.next = null;
                cp.next = null;
                return res;
            }
            Node nx2 = nx.next; // the next node of new list.
            cur.next = nx;
            cp.next = nx2;
            cur = cur.next;
        }  
        return res;
    }
}

class Solution { // Time/Space: O(n)
    public Node copyRandomList(Node head) {
        if (head == null) return null;
        Node cur = head;
        Map<Node, Node> map = new HashMap<>();
        // pass1: create map and copy nodes.
        while(cur != null){
            Node copy = new Node(cur.val);
            map.put(cur, copy);
            cur = cur.next;
        }        
        // pass2: wire 
        cur = head; // restore cur pointer to head.
        // map.get(cur);
        Node newHead = null;
        while(cur != null){            
            Node cp = map.get(cur); // get copy node
            if(newHead == null) newHead = cp; // set a head.
            Node nx = map.get(cur.next); // get copyed next node.
            Node rd = map.get(cur.random);
            cp.next = nx;
            cp.random = rd;
            cur = cur.next;
        }
        return newHead;
    }
}


// 863. All Nodes Distance K in Binary Tree
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
 
class Solution {  // Space : O(n) / Time: O(n) BFS m(edge)+n(node) -> m = n-1 
    public List<Integer> distanceK(TreeNode root, TreeNode target, int K) {
        List<Integer> res = new ArrayList<>();
     
        // map child -> parent
        Map<TreeNode, TreeNode> map = new HashMap<>();
        // only root's parent can be negetive.
        dfs(root, new TreeNode(-1), map);
        
        // for(Map.Entry<TreeNode, TreeNode> m : map.entrySet()){
        //     System.out.println(m.getKey().val + "=" + m.getValue().val );
        // }
        int level = 0;
        Queue<TreeNode> q = new LinkedList<>();
        Set<TreeNode> set = new HashSet<>();
        q.add(target);
        
        while(!q.isEmpty()){
            if(level == K){
                while(!q.isEmpty()){
                    res.add(q.poll().val);
                }
                break;
            }
            int size = q.size();
            for(int i = 0; i<size;i++){
                TreeNode cur = q.poll();
                set.add(cur);

                if (cur.left != null && !set.contains(cur.left)) q.offer(cur.left);
                if (cur.right != null && !set.contains(cur.right)) q.offer(cur.right);
                TreeNode parent = map.get(cur);
                if (parent.val != -1 && !set.contains(parent)) q.offer(parent);
            }
            level++;
        }        
        return res;
    }
    
    private void dfs(TreeNode root, TreeNode parent, Map<TreeNode, TreeNode> map){
        if(root == null) return;          
        map.put(root, parent);    
        if(root.left != null) dfs(root.left, root, map);
        if(root.right != null) dfs(root.right, root, map);
    }
}

// 735. Asteroid Collision / Time/ Space: O(n)
class Solution {
    public int[] asteroidCollision(int[] asteroids) {
        Stack<Integer> stack = new Stack<>();       
        for(int cur : asteroids){
            // push to stack directly if empty stack/ same sign/ <-, ->, turning away
            // [] / [-1,-2] / [1, 2] / [<- (-2, 2) ->]
            if(stack.isEmpty() || cur*stack.peek()>0 || (stack.peek()<0 && cur>0)){
                stack.push(cur);
            }else{ // in the case of collision
                Boolean pushCur = true;
                // [2, -2] / [5, -10] / [5, -1]
                while(!stack.isEmpty() && stack.peek()*cur < 0){
                    if (stack.peek() == -cur) {
                        stack.pop();
                        pushCur = false; // [2, -2] canceled out, don't push current val.
                        break; // no more to cancel. go out of the while loop.
                    }else if (stack.peek() < -cur){ // [5, -10]
                        stack.pop(); // continue to check on the stack.
                        // [-20,1,2,3,4,5,-10]
                    }else{
                        pushCur = false;
                        break; // no more to cancel.
                    }
                }
                if (pushCur) stack.push(cur);
            } 
        }  
        int [] res = new int[stack.size()];
        if(stack.isEmpty()) return res;
        // for(int i=0; i<stack.size(); i++){
        //     res[i] = stack.get(i);
        // }
        int i = res.length-1;
        while(!stack.isEmpty()){
            res[i--] = stack.pop();
        }       
        return res;
    }
}