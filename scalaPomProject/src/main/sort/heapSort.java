import java.util.Arrays;

/*
* 堆结构由两部分组成 ： heapInsert+heapify
* heapInsert是指形成大顶堆的过程，逐个元素添加进行，重新形成大顶堆
* heapify : 是指当大顶堆中有某个元素发生变化时，如何将它重新够造成大顶堆
*
* 构建一颗大顶堆的时间复杂度为log(N)
*
* 满二叉树与数组大小的关系为log(N)，即数组大小为N，二叉树的层数为log(N)
*
* 此时对于每一个数组元素军舰加进去，构建大根堆，此时的复杂度为：
* log1 + log2 + log3 + ... + logN-1
*
* 注意：大根堆在数组中并不存在，是我们脑补的结构
* 比如说：父节点=(i-1/)2
* */
public class heapSort{

    public static void hs(int[] arr){
        if (arr == null || arr.length<2){
            return;
        }
        for(int i = 0;i<arr.length;i++){
            heapInsert(arr,i);
        }
        // 形成大根堆后，将堆顶与末尾元素交换,从而将最大元素已经放在末尾
        int size = arr.length;
        swap(arr,0,--size);// 注意，堆顶是0号元素。
        while(size>0){
            heapify(arr,0,size);
            swap(arr,0,--size);
        }
    }

    public static void heapify(int[] arr,int start,int end){
        // 这主要是一个下沉的过程
        int left = start*2 + 1;
        while(left < end){// 是否沉到底部了
            // 找到堆顶元素下的两个子节点中最大值的下标
            int largest = (left + 1)<end && arr[left]<arr[left+1]?left+1:left;
            // 父节点与子节点比较
            largest = arr[start]>arr[largest] ? start : largest;
            if(largest==start){
                break;
            }
            swap(arr,start,largest);
            start = largest;
            left = start * 2 + 1;
        }
    }


    public static void heapInsert(int[] arr,int index){
        // 将当前index元素作为叶子节点加进来，找到它的父节点
//        int father = (int)(index-1)/2;
        while(arr[index]>arr[(index-1)/2]){
            swap(arr,index,(index-1)/2);
            index = (index-1)/2; // 当前节点上浮之后，继续找上一层父节点。
        }

    }

    public static void swap(int[] arr,int i,int j){
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    // for test
    public static void comparator(int[] arr) {
        Arrays.sort(arr);
    }

    // for test
    public static int[] generateRandomArray(int maxSize, int maxValue) {
        int[] arr = new int[(int) ((maxSize + 1) * Math.random())];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = (int) ((maxValue + 1) * Math.random()) - (int) (maxValue * Math.random());
        }
        return arr;
    }

    // for test
    public static int[] copyArray(int[] arr) {
        if (arr == null) {
            return null;
        }
        int[] res = new int[arr.length];
        for (int i = 0; i < arr.length; i++) {
            res[i] = arr[i];
        }
        return res;
    }

    // for test
    public static boolean isEqual(int[] arr1, int[] arr2) {
        if ((arr1 == null && arr2 != null) || (arr1 != null && arr2 == null)) {
            return false;
        }
        if (arr1 == null && arr2 == null) {
            return true;
        }
        if (arr1.length != arr2.length) {
            return false;
        }
        for (int i = 0; i < arr1.length; i++) {
            if (arr1[i] != arr2[i]) {
                return false;
            }
        }
        return true;
    }

    // for test
    public static void printArray(int[] arr) {
        if (arr == null) {
            return;
        }
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + " ");
        }
        System.out.println();
    }

    // for test
    public static void main(String[] args) {
        int testTime = 500000;
        int maxSize = 100;
        int maxValue = 100;
        boolean succeed = true;
        for (int i = 0; i < testTime; i++) {
            int[] arr1 = generateRandomArray(maxSize, maxValue);
            int[] arr2 = copyArray(arr1);
            hs(arr1);
            comparator(arr2);
            if (!isEqual(arr1, arr2)) {
                succeed = false;
                break;
            }
        }
        System.out.println(succeed ? "Nice!" : "Fucking fucked!");

        int[] arr = generateRandomArray(maxSize, maxValue);
        printArray(arr);
        hs(arr);
        printArray(arr);
    }
}