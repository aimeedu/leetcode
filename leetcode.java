
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
// or Time: O(n); Space O(n), where n is number of cells on the board, they are the same

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