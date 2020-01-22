
// One leetcode a day, keep unemployment away.

// 268. Missing Number
class Solution {
    public int missingNumber(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < nums.length; i++){
            n = n ^ i ^nums[i];
        }
        return n;
    }
}

class Solution {
    public int missingNumber(int[] nums) {
        Set<Integer> numSet = new HashSet<Integer>();
        for(int n: nums) numSet.add(n);
        //if no number is missing, total will be nums.length+1
        int total = nums.length+1;
        for (int n = 0; n<total; n++){
            if (!numSet.contains(n)){
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
        for(int i = 0; i<S.length(); i++){
            if(J.indexOf(S.charAt(i)) != -1){
                jew_count ++;
            }
        }
        return jew_count;
    }
}

// 169. Majority Element
// Approach 1: Brute Force
class Solution {
    public int majorityElement(int[] nums) {
        int majorityCount = nums.length/2;
        for(int num : nums){
            int count = 0;
            for(int ele : nums){
                if(ele == num){
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
        for(String s: strs){
            char[] sortedChar = s.toCharArray();
            Arrays.sort(sortedChar);
            String sorted = new String(sortedChar);
            if(map.get(sorted) == null){
                ArrayList<String> l = new ArrayList<>();
                l.add(s);
                map.put(sorted, l);
            }else{
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
        PriorityQueue<Integer> q = new PriorityQueue<>(k+1);
        for(int n: nums){
            q.add(n);
            if(q.size() > k){
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
        for(int i : nums) numsList.add(i);
    }

    public int add(int val) {
        numsList.add(val);
        Collections.sort(numsList);
        return numsList.get(numsList.size()-k);
    }
}

// 94. Binary Tree Inorder Traversal
class Solution {
    //declare arraylist must be global, or we need to use a helper function.
    List<Integer> l = new ArrayList<>();
    public List<Integer> inorderTraversal(TreeNode root) {
        if(root == null) return l;
        inorderTraversal(root.left);
        l.add(root.val);
        inorderTraversal(root.right);
        return l;
    }
}


// 01/10/2020

// 102. Binary Tree Level Order Traversal

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
    public List<List<Integer>> levelOrder(TreeNode root) {
        Queue<TreeNode> q = new LinkedList<>();
        List<List<Integer>> ans = new ArrayList<>();
        if (root == null) {
            return ans;
        }
        q.add(root);
        while(!q.isEmpty()){
            List<Integer> t = new ArrayList<>();
            // q.add(root);
            int size = q.size();
            for(int i = 0; i < size; i++){
                TreeNode c = q.remove();
                t.add(c.val);
                if(c.left != null) q.add(c.left);
                if(c.right != null) q.add(c.right);
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
        for (int card: hand){
            if(!map.containsKey(card)){
                map.put(card, 1);
            }else{
                map.put(card, map.get(card)+1);
            }
        }// put all the cards and it's frenquency in the TreeMap
        // Java.util.TreeMap uses a red-black tree in the background which makes sure that there are no duplicates;
        // HashMap implements Hashing, while TreeMap implements Red-Black Tree, a Self Balancing Binary Search Tree
        // additionally it also maintains the elements in a sorted order.
        System.out.println(hand.length);
        if (hand.length % W != 0) return false;
        //loop through the map for the smallest key
        while(map.size() > 0) {
            int first = map.firstKey();
            for (int k = first; k < first+W; k++){
                //when still in the loop but consecutive key does not exist, return false;
                if(!map.containsKey(k)) return false;
                //check the frequency count;
                int freq = map.get(k);
                // if the frequency == 1, remove this entry form count.
                if(freq == 1) map.remove(k);
                // decrease the frequency count.
                else map.put(k, freq-1);
            }
        }
        return true;
    }
}

// 01/11/2020

// 235. Lowest Common Ancestor of a Binary Search Tree
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
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(p.val<root.val &&q.val<root.val){
            return lowestCommonAncestor(root.left, p, q);
        }

        else if(p.val>root.val && q.val>root.val){
            return lowestCommonAncestor(root.right, p, q);
        }

        else return root;
    }
}

// 236. Lowest Common Ancestor of a Binary Tree
// Assume 2 nodes are both in the tree; this is important! So that you can avoid extra work!
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if( root == null )
            return null;
        if( root.val == p.val || root.val == q.val )
            return root;
        TreeNode left = lowestCommonAncestor( root.left, p, q);
        TreeNode right = lowestCommonAncestor( root.right, p, q);

        if( left != null && right != null )
            return root;
        if( left == null && right == null )
            return null;
        return left != null ? left : right;
    }
}

// 108. Convert Sorted Array to Binary Search Tree
class Solution {
    public TreeNode sortedArrayToBST(int[] nums) {
        return sol(nums,0,nums.length - 1);
    }
    public static TreeNode sol(int[] nums, int start, int end){

        if(start > end ) return null;
        // int mid = ((start + end)/2);
        TreeNode root = new TreeNode(nums[((start + end)/2)]);
        root.left = sol(nums,start, ((start + end)/2) -1);
        root.right = sol(nums,((start + end)/2) + 1, end);
        return root;

    }
}

// 01/12/2020

// 105. Construct Binary Tree from Preorder and Inorder Traversal
class Solution {

    Map<Integer, Integer> map;

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if(preorder == null || inorder == null || preorder.length == 0 || preorder.length != inorder.length) return null;

        map = new HashMap<>();

        for(int i = 0; i< inorder.length; i++){
            map.put(inorder[i], i);
        }

        return build(preorder, inorder, 0, 0, preorder.length-1);
    }

    public TreeNode build(int[] preorder, int[] inorder, int pre_st, int in_st, int in_end){
        // in_st = 0, fixed value, when passing as parameter in the recursive function. not apply to right subtree recursion.
        // in this function, we keep checking the indexes in both arrays, without recreating new subarrays, this approach is time efficient.
        if(pre_st > preorder.length || in_st>in_end) return null;
        System.out.println(preorder[pre_st]);
        TreeNode root = new TreeNode(preorder[pre_st]);
        // set root_index initial to the start of the inorder array,

        // in preorder, the root always goes first, so the next element in preorder array is the root we need to locate in inorder array.
        // to find the the root in inorder array, we need to check inorder[root_index] == preorder[pre_st] ?
        // so we can locate the current root_index in inorder array.
        // loop is not efficient, we use hashmap.
        // while(root_index<=in_end && inorder[root_index] != preorder[pre_st]) {
        //     root_index++;
        // }

        int root_index = map.get(preorder[pre_st]); //pass key get value.

        root.left = build(preorder, inorder, pre_st+1, in_st, root_index-1);
        // when we finish the left subtree,
        // for right sub tree, the pre_st = pre_st+(size of left subtree) inthe form of array.
        // the size of left subtree = the size to the left of the root in inorder array.
        root.right = build(preorder, inorder, pre_st+(root_index-in_st)+1, root_index+1, in_end);

        return root;
    }
}

// 103. Binary Tree Zigzag Level Order Traversal
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        Stack<TreeNode> s1 = new Stack<>(); //
        Stack<TreeNode> s2 = new Stack<>();

        List<List<Integer>> ans = new ArrayList<>();

        if (root == null) return ans;

        s1.add(root); // initial state, add root to s1.

        while(!s1.isEmpty() || !s2.isEmpty()){
            List<Integer> a1 = new ArrayList<>();
            while(!s1.isEmpty()){
                TreeNode t = s1.pop();
                a1.add(t.val);
                if(t.left != null) s2.push(t.left);
                if(t.right != null) s2.push(t.right);
            }
            if(!a1.isEmpty()) ans.add(a1);

            List<Integer> a2 = new ArrayList<>();
            while(!s2.isEmpty()){
                TreeNode t = s2.pop();
                a2.add(t.val);
                if(t.right != null) s1.push(t.right);
                if(t.left != null) s1.push(t.left);
            }
            if(!a2.isEmpty()) ans.add(a2);
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
        if (root == null) return null;
        if (root.left != null && root.right != null){
            root.left.next = root.right;
        }
        if(root.next != null && root.right != null) {
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
        if(root==null) return true;
        if (root.val <= min ||root.val >= max) return false;
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
        int dp[] = new int[n+1];
        dp[0] = 1;
        dp[1] = 1;
        for(int i = 2; i < n+1; i++){
            dp[i] = dp[i-1] + dp[i-2];
        }
        return dp[n];
    }
}

// 322. Coin Change
class Solution {
    public int coinChange(int[] coins, int amount) {
        int [] dp = new int[amount+1];
        Arrays.fill(dp, amount+1);
        // fill the dp array with amount+1. since # of ways is as much as amount.
        dp[0] = 0;
        // int ans = 0;
        for(int i=0; i<dp.length; i++){
            for(int c=0; c<coins.length; c++){
                if(i >= coins[c]){
                    dp[i] = Math.min(dp[i-coins[c]]+1, dp[i]);
                }
            }// end loop coins array
        }// end finding min for dp[i];
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
        if(text1.length() == 0 || text2.length() == 0 || text1.isEmpty() || text2.isEmpty())
            return 0;

        // take the substring without the last character.
        String s1 = text1.substring(0, text1.length()-1);
        System.out.println(s1);
        String s2 = text2.substring(0, text2.length()-1);
        System.out.println(s2);
        // get the last character for both strings;
        if(text1.charAt(text1.length()-1) == text2.charAt(text2.length()-1)){
            return 1 + longestCommonSubsequence(s1, s2);
        }
        else{
            return Math.max(longestCommonSubsequence(text1, s2), longestCommonSubsequence(s1, text2));
        }
    }
}
// bottom-up  DP Solution
// runtime (O(n*m))
class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
        int n = text1.length();
        int m = text2.length();

        int[][] dp = new int[n+1][m+1];
        // length() +1, because we need too consider the empty string.

        for(int i=0; i<= n; i++){
            for(int j=0; j<= m; j++){

                // fill the 2d array with 0 , when either string is empty.
                if(i==0 || j==0)
                    dp[i][j] = 0;

                else if(text1.charAt(i-1) == text2.charAt(j-1)){
                    dp[i][j] = dp[i-1][j-1] + 1;
                }
                else{
                    dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
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
        if(s.length() == 0) return true;
        if(s.length() == 0 && t.length() == 0) return true;
        if(s.length() != 0 && t.length() == 0) return false;
        if(s.length() > t.length()) return false;

        int n = s.length();
        int m = t.length();
        // dp table include empty string.
        boolean [][] dp = new boolean[n+1][m+1];

        for (int i = 0; i<= n; i++) {
            for (int j = 0; j<= m; j++){
                if(i == 0) dp[i][j] = true;
                else if(j == 0 && i != 0 ) dp[i][j] = false;
                else if(s.charAt(i-1) == t.charAt(j-1)){
                    dp[i][j] = true && dp[i-1][j-1];
                }
                else{
                    dp[i][j] = dp[i-1][j] && dp[i][j-1];
                }
            }
        }
        return dp[n][m];
    }
}
// O(n)
class Solution {
    public boolean isSubsequence(String s, String t) {
        for(int i = 0; i<s.length(); i++){
            int index = t.indexOf(s.charAt(i));
            // check if character in s is also in t?
            if(index >= 0) t=t.substring(index+1);
                else return false;
        }
        return true;
    }
}

// 53. Maximum Subarray
// Time O(n) Space O(1)
class Solution {
    public int maxSubArray(int[] nums) {
        int maxSum = nums[0];
        for(int i = 1; i<nums.length; i++){
            nums[i] = Math.max(nums[i], nums[i-1]+nums[i]);
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
        if (m<0 || n<0) return 0;
        if (m == 1 && n == 1) return 1;
        int[][] path = new int[m+1][n+1];
        if (path[m][n] != 0) return path[m][n];
        int left = uniquePaths(m-1, n);
        int up = uniquePaths(m, n-1);
        path[m][n] = left + up;
        return path[m][n];
    }
// DP Solution O(nm)
class Solution {
    public int uniquePaths(int r, int c) { //3(row) * 2(col) == 2(row) * 3(col)
        if(r==0 ||c==0) return 0;
        int[][] path = new int[r][c];

        for (int i=0; i< r; i++){
            for (int j=0; j< c;j++){
                if(i == 0 || j == 0) {
                      path[i][j]=1;
                }
                else{
                    path[i][j] = path[i-1][j] + path[i][j-1];
                }
            }
        }
        return path[r-1][c-1];
    }
}

// 279. Perfect Squares similar to 322. Coin Change

class Solution {
    public int numSquares(int n) {
        // [1, 2, 3, 4, 5, 6, 7, ...]^2
        // [1, 4, 9, 16, 25, 36, 49, ...]
        double rt = Math.sqrt(n);
        int anchor =  (int)Math.floor(rt); // largest sqrt.

        int [] dp = new int [n+1]; // [13,13,13,13,...]
        Arrays.fill(dp, n+1);
        dp[0] = 0;

        int [] a = new int [anchor]; // array to store perfect square. [1, 4, 9, 16, 25, 36, 49, ...]
        for (int i = 0; i < anchor; ++i) {
            a[i] = (i+1) * (i+1);

        }

        for(int i=0; i<dp.length; i++){ // i repersents the subcase n.
            for(int j=0; j<a.length; j++){
                if(i >= a[j]) { // the number n must > than the perfect square.
                    dp[i] = Math.min(dp[i], dp[i-a[j]]+1);
                }
            }
        }
        return dp[n];
    }
}
// better
class Solution {
    public int numSquares(int n)
    {
        int dp[]=new int[n+1];
        dp[0]=0;
        for(int i=1;i<=n;i++)
        {
            int min=Integer.MAX_VALUE;
            for(int j=1;j*j<=i;j++)
            {
                if(i-j*j>=0)
                    min=Math.min(min,dp[i-j*j]);
            }
            dp[i]=min+1;
        }
        return dp[n];
    }
}
