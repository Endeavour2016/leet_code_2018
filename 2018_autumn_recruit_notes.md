##  一 数据库
### 1.1 数据库并发一致性
#### 脏读

一个事务读取了另外一个事务还没有提交的数据。比如 T1 修改了一个数据，T2 读取，随后 T1 撤销了修改，此时 T2 读取的数据即为脏数据。
**解决脏读的方法：**把数据库的事务隔离级别调整到 Read_committed

#### 不可重复读

在同一个事务内，对相同的数据进行多次读取得到的结果不同
不可重复读的特点是：一个事务内读取数据，还没有结束时，另一个事务对数据进行了修改；这样第一个事务再次读取数据时，结果与第一次不一致
**举例：** 在事务1中，mary 读取了自己的工资为1000，操作并没有完成；在事务2中，财务人员修改了mary 的工资为 2000，并提交了事务; 在事务1中再次读取工资，变为2000。前后读取结果不一致！

#### 幻读

T1读取某个范围的数据， T2在这个范围内插入新的数据，T1再次读取这个范围的数据，此时读取的结果和第一次读取的结果不同。

**解决方法：**把数据库的隔离级别调整到 Repeatable_read

“脏读”、“不可重复读”和“幻读”，其实都是数据库读一致性问题。

级别高低：脏读 < 不可重复读 < 幻读



### 1.2 事务的隔离级别

#### 未提交读（READ UNCOMMITTED）：最低级别

事务中的修改，即使没有提交，对其它事务也是可见的。
会出现脏读现象。



#### 提交读（READ COMMITTED）：语句级

一个事务只能读取已经提交的事务所做的修改。换句话说，一个事务所做的修改在提交之前对其它事务是不可见的。当隔离级别设置为 read committed 时，避免了脏读，但是可能会造成不可重复读。



#### 可重复读（REPEATABLE READ）：事务级

保证在同一个事务中多次读取同样数据的结果是一样的。



#### 可串行化（SERIALIZABLE）：最高级别

Serializable 性能很低，一般很少使用，在该级别下，事务按照顺序执行，可以避免脏读、不可重复读、幻读。



**不可重复读与幻读的区别**
不可重复读是针对其它事务修改了正在被读取的数据(UPDATE)，幻读是针对其它事务向表中新增(删除)了一条记录（INSERT）

**MYSQL 中隔离级别**

  - 隔离级别最高的是 Serializable 级别，最低的是 Read uncommitted 级别。级别越高，执行效率就越低。像 Serializable 这样的级别，就是以锁表的方式(类似于Java多线程中的锁)使得其他的事务只能等待.  

  - 在 MySQL 数据库中，支持上面四种隔离级别，默认的为 Repeatable read (可重复读)；而在 Oracle 数据库中，只支持 Serializable (串行化)级别和 Read committed (读已提交)这两种级别，其中默认的为 Read committed 级别。

### 1.3 MyISAM 与 Innodb 存储引擎的对比 

- MyISAM 类型不支持事务处理等高级处理，而InnoDB类型支持。
- MyISAM 表不支持外键，InnoDB 支持
- MyISAM 锁的粒度是表级，而 InnoDB 支持行级锁
- MyISAM 支持全文类型索引，而 InnoDB 不支持全文索引。(mysql 5.6后innodb支持全文索引)
- MyISAM 相对简单，所以在效率上要优于 InnoDB，小型应用可以考虑使用 MyISAM。
- 当你的数据库有大量的写入、更新操作而查询比较少或者数据完整性要求比较高的时候就选择 innodb 表。当你的数据库主要以查询为主，相比较而言更新和写 入比较少，并且业务方面数据完整性要求不那么严格，就选择 mysiam 表。



### 1.4 数据库的范式

- 第一范式（1NF）：**列的原子性**，即每一列不可再分割。
  考虑这样一个表：【联系人】（姓名，性别，电话）。如果在实际场景中，一个联系人有家庭电话和公司电话，那么这种表结构设计就没有达到 1NF。要符合 1NF 我们只需把列（电话）拆分，即：【联系人】（姓名，性别，家庭电话，公司电话）。1NF 很好辨别，但是 2NF 和 3NF 就容易搞混淆。  

- 第二范式（2NF）：首先满足 1NF。
  此外，表必须有一个主键；非主键的列列必须**完全依赖于主键**，而不能只依赖于**主键的一部分**。不满足第二范式的表：学号、姓名、年龄、课程名称、成绩、学号。
**可能会出现的问题**
	- 数据冗余：每条记录含有相同的信息
	- 更新异常：调整课程学分，所有行都调整
	- 删除异常：删除所有学生成绩，把课程信息全都删除了
**正确做法**
	- 学生表：student(学号，姓名，年龄)
	- 课程表：course(课程名称，学分)
	- 选课关系：studentCourse(学号，课程名称，成绩)

- 第三范式（3NF)：首先满足 2NF。
  另外非主键列必须**直接依赖于主键**，不能存在**传递依赖**。即不能存在：非主键列 A 依赖于非主键列 B，非主键列 B 依赖于主键的情况。 



### 1.5 事务的四大特性：ACID

- 原子性（atomicity）：事务中的操作要么全部执行，要么全部失败回滚
- 一致性（consistency）：事务执行前后，数据库的状态都满足完整性约束
- 隔离性（isolation）：多个事务并发执行，一个事务不会受到其他事务干扰
- 持久性（durality）：事务执行后，对数据的修改是永久性的，即使数据库崩溃，所做的修改也不会改变。



### 1.6 数据库锁机制

#### 概述

- MyISAM 支持表锁，InnoDB 支持表锁和行锁，默认为行锁
- 表级锁：开销小，加锁快，不会出现死锁（deadlock free）。锁定粒度大，发生锁冲突的概率最高，并发度最低
- 行级锁：开销大，加锁慢，会出现死锁。锁力度小，发生锁冲突的概率小，并发度最高。

#### 锁的分类

从对数据库操作的类型分，分为读锁和写锁 

- 读锁（共享锁）：针对同一份数据，多个读操作可以同时进行而不会互相影响 
- 写锁（排它锁）：当前写操作没有完成前，它会阻断其他写锁和读锁（即阻塞其他的写和读请求）
- 从对数据操作的粒度分，分为表锁和行锁。

#### MySQL 表级锁模式（MyISAM)

- 表共享锁（Table Read Lock）和表独占写锁（Table Write Lock）
- 读读共享，读写互斥，写写互斥，写读互斥（比如写读互斥，当一个用户对user表进行写的时候，会阻塞其他用户的写操作和读操作）


- 加表锁：
  - 在 select 的时候自动加读锁，在 update/delete/insert时，自动加写锁
  - 一般不需要用户用lock table 命令直接显式加锁
  - 显示加锁：lock table tablename read;

- 表锁的调度：
  - 当一个进程请求user表的读锁，一个进程请求user表的写锁，即使读请求先到，也是先满足写锁，即写锁的优先级更高！
  - 所以，MyISAM 不适合大量的更新，因为大量更新会导致查询操作很难获得读锁

#### InnoDB行级锁

共享锁（s）
排他锁（Ｘ）

另外，为了允许行锁和表锁共存，实现多粒度锁机制，InnoDB 还有两种内部使用的意向锁（Intention Locks），这两种意向锁都是表锁。

- 意向共享锁（IS）：事务打算给数据行加共享锁，事务在给一个数据行加共享锁前必须先取得该表的IS锁。
- 意向排他锁（IX）：事务打算给数据行加排他锁，事务在给一个数据行加排他锁前必须先取得该表的IX锁。



#### SQL中显示加锁

- 共享锁（Ｓ）：SELECT * FROM table_name WHERE ... LOCK IN SHARE MODE
- 排他锁（X）：SELECT * FROM table_name WHERE ... FOR UPDATE。 


- **加锁方式**

  意向锁是InnoDB自动加的
  对于update insert delete 会自动加排它锁
  对于 select 不会自动加锁
  Select * from user LOCK IN SHARE MODE  手动添加共享锁(行锁)
  select * from user For update 手动添加排它锁（行锁）


- **行锁的实现**

  通过索引条件检索数据时，才使用行锁，否则用表锁
  也就是如果select的where列不是索引列，那么此时，实际上是给整个表加锁
  检索条件是索引时，又分索引是聚集索引还是非聚集索引，如果是聚集索引（索引上叶节点存放的整条记录），在聚集索引记录上加上行锁；而如果是非聚集索引，既要在非聚集索引记录上加行锁，又要在聚集索引上加行锁。

**什么时候使用表级锁？**

- 事务需要更新表的大部分数据。如果每一行都加上排他锁，锁冲突严重，那么此时执行效率低
- 事务涉及多个表，比较复杂。



## 二 数据结构与算法

#### 分治法
基本思想是将原问题划分为若干个规模较小的独立子问题，递归的解决这些子问题，然后合并这些问题的结果

- 分解（Divide）：将原问题分解成一系列子问题
- 解决（conquer）：递归地解决各个子问题。若子问题足够小，则直接求解
- 合并（Combine）：将子问题的结果合并成原问题的解
归并排序和快速排序就是典型的分治法的例子！

#### 动态规划

动态规划适用于子问题有重叠的情况，若直接采用分治法会进行很多不必要的操作，即重复的求解公共子问题。动态规划对每个子问题求解一次，结果存放在一张表中，从而避免每次遇到子问题重复求解。  

**两个基本要素**
- 最优子结构：一个问题的最优解中包含了子问题的最优解，则该问题具有最优子结构

- 重叠子问题

**求解步骤**

- 列出递推公式
- 自底向上逐步求解子问题的解



## 三 算法思路题

面试中针对一些特定的应用场景，设计算法或解决思路！

1、给定a、b两个文件，各存放 50 亿个 url，每个 url 各占 64 字节，内存限制是 4G，让你找出 a、b 文件共同的 url?

第一步：遍历文件 a，使用 Hash 函数将a文件中的url分别存储到1000个小文件中，这样每个小文件的大约为300M；遍历文件 b，使用相同的 Hash 函数，将每个url存储到1000个小文件中。这样，所有可能相同的 url 都存在对应的小文件中。

第二步：求每对小文件中相同的 url，可以把其中一个小文件的 url 存储到 hash 表中。然后遍历另一个小文件的每个 url，看其是否在刚才构建的 hash 表中，如果是，那么就是共同的 url，存到文件里面即可。  



## 四 Linux常用命令

**进程相关的命令**
- ps：现实当前运行的进程的状态。常用方式 ps -ef 或 ps aux
    - -e：现实所有的进程
    - -f：全格式现实
    - -h：不显示标题
    - -l：长格式
    - u：以用户为主的格式来显示程序状况
    - x：显示所有的程序
- 查看文本的命令
    - tail：从后显示文件的内容
    - head：显示文件的前几行
    - more：一页一页的显示文件内容
    - cat：显示文件内容或者将几个文件连接起来显示
	cat 主要有三大功能：
    1.一次显示整个文件：cat filename
    2.从键盘创建一个文件：cat > filename 只能创建新文件,不能编辑已有文件.
    3.将几个文件合并为一个文件：cat file1 file2 > file

**系统管理有关的命令**
- free 命令：查看系统内存的使用情况，使用free -h以G为单位显示内存

- netstat 命令

  用于查看网络状态  



## 五 操作系统

#### 进程与线程的区别

- 进程是程序的一次执行过程，系统资源分配（CPU时间、内存等）的基本单位。
- 线程是进程的一个执行流（或者说是进程中的不同执行路径），是 CPU 调度和分派的基本单位，它是比进程更小的能独立运行的基本单位。一个进程由几个线程组成（拥有很多相对独立的执行流的用户程序共享应用程序的大部分数据结构），线程与同属一个进程的其他的线程共享进程所拥有的全部资源。
- 进程有独立的地址空间，线程没有单独的地址空间（同一进程内的线程共享进程的地址空间）；但是线程有自己的堆栈和局部变量。
- 一个进程崩溃后，通常不会影响其他进程。一个线程崩溃导致整个进程崩溃
- 多进程程序健壮性更强，但是进程切换时，耗费资源较大，效率相对较低！

#### Linux/Unix线程同步的方法

- 互斥量 mutex
- 读写锁
- 条件变量
- 自旋锁
- 屏障

#### 死锁的四个必要条件

- 互斥条件
- 占有和等待条件
- 不可抢占条件
- 环路等待条件

#### 死锁的避免

**银行家算法**

  - 死锁避免的基本思想：系统对进程的**资源申请进行动态的检查**，如果分配资源后系统进入不安全状态，则不予分配；若能够进入安全状态，则分配资源，这是一种保证系统不进入死锁状态的动态策略。 

  - 安全状态：如果能找到一个进程推进的序列 P1, P2, P3, P4, …, Pn。对于其中每个进程 Pi， 都能够满足其最大的资源需求量，则系统处于安全状态

1. LRU 缓存的实现 C++



#### 几种缓存策略

  - LRU (Least Recently Used，最近最少使用)。如果数据近期很少被被访问，则将来被访问的几率也会很低。当有新数据进来时，若缓存已满，则近期最少被使用的数据会被淘汰掉。
  - FIFO (Fist in first out, 先进先出)。数据按照到达的先后次序存放。 如果一个数据最先进入缓存中，则应该最早淘汰掉。（实际上最先进来的数据也有可能是最近访问的，LRU 算法中会把此数据移动到链表首部，表明这是最近访问的，这样该数据不会被淘汰）。

**LRU缓存的实现思路**
  - 新数据或击中的数据放到链表头部head，表示最近使用的数据。（数据命中指的是访问cache中的数据，访问后把数据移动到链表 head 处）
  - 如果链表满，从尾部淘汰数据。
  - 但只用链表会存在一个问题，命中数据的时间复杂度为O(n)，每次需要遍历链表。所以引入哈希表，快速命中其中的数据（时间复杂度降到O(1)，以空间换时间）。

**LRU缓存所具备的操作**
  - set(key, value)：设置某个元素的值。如果 key 在 hash 表中存在，则先重置对应的value值，然后获取对应的节点 cur，将 cur 节点从链表删除，并移动到链表的头部；若果 key 在 hash 表中不存在，则新建一个节点，并将节点放到链表的头部。当 Cache 存满的时候，将链表的尾部节点删除即可。
  - get(key)：访问元素。如果 key 在 hash 表存在，则把对应的节点放到链表头部，并返回对应的 value 值；如果不存在，则返回 -1。

**C++代码实现**
  - 利用STL中list存放数据，每个节点中存放的元素为（key,value）, 用结构体表示。

  - 利用 unordered_map 存放存放 key,便于快速命中元素。

```c++
// 节点中存放到元素
struct Element {
    int key;
    int value;
    Element(int k, int v):key(k), value(v){}
};
```
定义一个Cache类，里面封装各种数据及接口函数
```c++
class LRUCache {
private:
    list<Element> m_list;
    unordered_map<int, list<Element>::iterator> m_map;
    int m_capacity;
public:
    LRUCache(int capacity) {
        m_capacity = capacity;
    }

    // 查找指定的 key
    int get(int key) {
        if (m_map.find(key) == m_map.end()) // cache中没有key，直接返回
            return -1;
        else {
            //将元素移动到链表头部
            //splice(begin(), src_list&, iterator);
            m_list.splice(m_list.begin(), m_list, m_map[key]); 
            // 把 m_list 中的 m_map[key]
            // 放到 m_list 中的首部，原位置中的 m_map[key] 则被删除
            m_map[key] = m_list.begin();
            return m_list.begin()->value;
        }
    }
    // 放入元素
    void put(int key, int value) {
        assert(m_capacity > 0);
        if (m_map.find(key) != m_map.end())
        {   // 如果已经存在key, 则更新 value
            m_map[key]->value = value;
            //将元素放入链表头部
            m_list.splice(m_list.begin(), m_list, m_map[key]);
            m_map[key] = m_list.begin();
        }
        else if (m_capacity == m_list.size()) // cache容量已满
        {
            m_map.erase(m_list.back().key); // 删除链表尾部的key
            m_list.pop_back();
            // 在链表首部插入元素
            m_list.push_front(Element(key, value));
            m_map[key] = m_list.begin();
        }
        else // cache尚未存满，直接插入到首部
        {
            m_list.push_front(Element(key, value));
            m_map[key] = m_list.begin();
        }
    }
};
```



## 六 设计模式

#### 单例模式

保证一个类仅有一个实例，并提供一个访问它的全局访问点，该实例被所有程序模块共享。

1、懒汉模式：第一次使用实例时才构造对象实例

```c++
class singleton //实现单例模式的类
{
private:
	singleton(){} //私有的构造函数
	static singleton* Instance;
public:
	static singleton* GetInstance() {
		if (Instance == NULL) //判断是否第一调用
			Instance = new singleton();
		return Instance;
	}
};
```

2、改进的懒汉模式（使用局部静态变量，线程安全版本）

```c++
class singleton   //实现单例模式的类
{
private:
	singleton() {}  //私有的构造函数
public:
	static singleton* GetInstance() {
		static singleton Instance;
		return &Instance;
	}
};
```

3、懒汉模式：双重加锁机制

```c++
class singleton   //实现单例模式的类
{
private:
	singleton(){}  //私有的构造函数
	static singleton* Instance;
public:
	static singleton* GetInstance() {
		if (Instance == NULL){ //判断是否第一调用 
			Lock(); //表示上锁的函数（pthread_mutex_lock）
			if (Instance == NULL){
				Instance = new singleton();
			}
			UnLock() //解锁函数
		}			
		return Instance;
	}
};
```

4、饿汉式：在类中直接创建了静态实例对象，调用函数时直接返回该实例对象

```c++
class singleton {   //实现单例模式的类
private:
	singleton(){}  //私有的构造函数
	static singleton* Instance;
public:
	static singleton* GetInstance() {
		return Instance;
	}
};

singleton* singleton::Instance = new singleton(); // 静态变量在类外初始化
```



## 七 C++ basic

### 堆和栈的区别

- 分配和管理方式不同
  - 堆的空间是程序员手动分配和释放的，是动态的分配
  - 栈的空间是由编译器自动分配和释放的，比如局部变量的分配。
- 产生碎片不同
  - 频繁的使用new和malloc分配内存，会造成内存空间的不连续，产生大量碎片，降低程序效率；
  - 对于栈，则不存在碎片问题，原因是栈是先进后出的，不可能有一个内存块从中间弹出来。
- 生长方向不同
  - 堆是朝着内存地址增加的方向增长；
  - 栈是朝着地址减小的方向增长；

### C++11新特性

1、类型推导：auto, decltype
```C++
auto i = 5;
auto it = new int(10);
decltype(表达式)；
```

2、范围 for 循环，类似 java 的for循环

3、初始化列表使用更加广泛

4、lambda 表达式：匿名函数机制，基本语法如下

```c++
[capture] (para) options->return type {body; };

capture是捕获列表； 
params是参数表；(选填) 
options 是函数选项；可以填 mutable, exception, attribute（选填） 
mutable 说明 lambda 表达式体内的代码可以修改被捕获的变量，并且可以访问被捕获的对象的non-const 方法；默认情况下无法修改被捕获的变量。exception 说明 lambda 表达式是否抛出异常以及何种异常。attribute 用来声明属性。 
return type 是返回值类型（拖尾返回类型）。(选填)
body是函数体。

// lambda表达式捕获类型
[]   // 不捕获任何外部变量
[=]  // 以值的形式捕获所有外部变量
[&]  // 以引用形式捕获所有外部变量
[x, &y]  // x 以传值形式捕获，y 以引用形式捕获 
[=, &z]  // z 以引用形式捕获，其余变量以传值形式捕获
[&, x]   // x 以值的形式捕获，其余变量以引用形式捕获

```

#### 右值引用与移动构造

移动构造把旧对象的资源直接转移给新对象，可以减少不必要的内存复制，带来性能上的提升。

```C++
A(A&& rhs) : _ptr(rhs.ptr) {
    rhs.ptr = nullptr;
}
```



### 智能指针原理及实现

#### 指针指针的概念

智能指针实际上是一个类，内部对普通的指针进行了封装。定义了智能指针对象后，其行为表现的像是一个指针。
智能指针类的构造函数传入一个普通指针，析构函数中释放传入的指针；

C++11 引入了新的智能针:  shared_ptr, weak_ptr,  unique_ptr；包含在头文件 <memory> 中
  shared_ptr维护了一个指向 control block 的指针，用于记录所指内存的引用个数。    



#### 智能指针用法

```c++    
/** shared_ptr, 最常用的智能指针
* 1. 基于引用计数的智能指针。其内部的引用计数为 0 时，指向的内存会被释放。
* 2. shared_ptr维护了一个指向 control block 的指针，用于记录所指内存的引用个数。
* 3. shared_ptr 内部的引用计数是线程安全的，但是对象的读取需要加锁。
* 4. 注意避免循环引用, shared_ptr 的一个最大的陷阱是循环引用，
* 5. 循环引用会导致堆内存无法正确释放，导致内存泄漏。
* 6. 赋值时，比如 ptr1 = ptr2; 使得 ptr1 原来所指内存的引用计数减 1，当计数 为 0 时，自动释放内存
而 ptr2 所指向的内存引用计数加 1。
*/

// shared_ptr
int a = 10;
shared_ptr<int> sp1 = make_shared<int>(a);
shared_ptr<int> sp2(new int(18));
cout << sp2.use_count() << endl;
sp1.get(); // 获取原始指针

// unique_ptr
unique_ptr<int> p1(new int(5));
// unique_ptr<int> p2 = p1; // 编译会出错
// 转移所有权, 现在那块内存归p3所有, p1成为无效的指针.
unique_ptr<int> p3 = std::move(p1); 
```



#### shared_ptr 的简单实现

```c++
#include <iostream>
using namespace std;

template<class T>
class SmartPtr
{
public:
    SmartPtr(T *p);
    ~SmartPtr();
    SmartPtr(const SmartPtr<T> &orig);                // 浅拷贝
    SmartPtr<T>& operator=(const SmartPtr<T> &rhs);    // 浅拷贝
    T* operator->();
    T& operator*();
private:
    T *ptr;
    // 将 use_count 声明成指针是为了方便对其的递增或递减操作
    int *use_count; // 实际的shared_ptr中，该指针指向一个control block，里面存放的引用计数值
};

template<typename T>
SmartPtr<T>::SmartPtr(T *p) : ptr(p)
{
    try {
        use_count = new int(1);
    }
    catch (bad_alloc) {
        delete ptr;
        ptr = nullptr;
        use_count = nullptr;
        cout << "Allocate memory for use_count fails." << endl;
        exit(1);
    }

    cout << "Constructor is called!" << endl;
}

template<class T>
SmartPtr<T>::~SmartPtr()
{
    // 只在最后一个指向对象的shared_ptr销毁时，才释放对象的内存
    if (--(*use_count) == 0)
    {
        delete ptr; // 销毁ptr指向的内存
        delete use_count;

        ptr = nullptr;
        use_count = nullptr;
        
        cout << "Destructor is called!" << endl;
    }
}

template<class T>
SmartPtr<T>::SmartPtr(const SmartPtr<T> &orig)
{
    if (this != &orig) { // 相同的元素没必要进行拷贝初始化
        ptr = orig.ptr;
        use_count = orig.use_count;
        ++(*use_count);
        cout << "Copy constructor is called!" << endl;
    }
}
// 重载等号函数不同于复制构造函数，即等号左边的对象可能已经指向某块内存。
// 这样，我们就得先判断左边对象指向的内存已经被引用的次数。如果次数为1，
// 表明我们可以释放这块内存；反之则不释放，由其他对象来释放。
template<class T>
SmartPtr<T>& SmartPtr<T>::operator=(const SmartPtr<T> &rhs)
{
    // 《C++ primer》：“这个赋值操作符在减少左操作数的使用计数之前使rhs的使用计数加1，
    // 从而防止自身赋值”而导致的提早释放内存
    ++(*rhs.use_count);
    // 将左操作数对象的使用计数减1，若该对象的使用计数减至0，则删除该对象
    if (--(*use_count) == 0) {
        delete ptr;
        delete use_count;
        cout << "Left side object is deleted!" << endl;
    }
    ptr = rhs.ptr;
    use_count = rhs.use_count;
    
    cout << "Assignment operator overloaded is called!" << endl;
    return *this;
}

template<typename T>
T& SmartPtr<T>::operator*() {
    return *ptr;
}

template<typename T>
T* SmartPtr<T>::operator->() {
    return ptr;
}
```



#### unique_ptr, shared_ptr, weak_ptr 特点小结：

- unique_ptr独享被管理对象，同一时刻只能有一个unique_ptr拥有对象的所有权，当其被销毁时被管理对象也自动被销毁。无法进行拷贝，只能移动move。
- shared_ptr共享被管理对象，同一时刻可以有多个shared_ptr指向对象内存；当最后一个shared_ptr对象销毁时，被管理对象自动销毁
- weak_ptr用于指向shared_ptr所管理的对象，但是本身并不占用引用计数。它可以判断对象是否存在和返回指向对象的shared_ptr类型指针；用途之一解决shared_ptr循环引用问题。



#### 智能指针注意事项

1、不要把一个原生指针给多个shared_ptr或者unique_ptr管理
```C++
int* ptr = new int;
shared_ptr<int> p1(ptr);
shared_ptr<int> p2(ptr); 
//p1,p2析构的时候都会释放ptr，同一内存被释放多次！
```

2、如果不是通过new得到的动态资源内存请自定义删除器（如malloc分配的资源）

3、尽量使用make_shared,不要把原生指针暴露出来

4、shared_ptr可能会出现循环引用的情况，此时用weak_ptr可以解决此类问题



## 八 计算机网络

#### TCP 拥塞控制

- TCP的拥塞控制采用的是窗口机制，通过调节窗口的大小实现对数据发送速率的调整。
- TCP的发送端维持一个称为拥塞窗口cwnd的变量，单位为字节，用于表示在未收到接收端确认的情况下，可以连续发送的数据字节数。
- cwnd的大小取决于网络的拥塞程度，并且动态地发生变化。
- 拥塞窗口调整的原则是：只要网络没有出现拥塞，就可以增大拥塞窗口，以便将更多的数据发送出去，相当于提高发送速率；一旦网络出现拥塞，拥塞窗口就减小一些，减少注入网络的数据量，从而缓解网络的拥塞。
- 发送端判断网络发生拥塞的依据是：发送端设置一个重传计时器 RTO，对于某个已发出的数据报文段，如果在 RTO 计时到期后，还没有收到来自接收端的确认，则认为此时网络发生了拥塞。

#### TCP 拥塞控制算法

- 慢启动
- 拥塞避免
- 快重传
- 快恢复



#### TCP 拥塞控制过程

- TCP连接初始化，将拥塞窗口设置为1
- 执行慢开始算法，cwd 按指数规律增长，知道 cwd == ssthress 开始执行拥塞避免算法，cwnd 按线性规律增长
- 当网络发生拥塞，把 ssthresh 值更新为拥塞前 ssthresh 值的一半，cwnd 重新设置为1，按照步骤（2）执行。
- 快重传算法并非取消了重传机制，只是在某些情况下更早的重传丢失的报文段（如果当发送端接收到三个重复的确认ACK时，则断定分组丢失，立即重传丢失的报文段，而不必等待重传计时器超时）。慢开始算法只是在 TCP 建立时才使用。  


 **快恢复算法有以下两个要点：**

- 当发送方连续收到三个重复确认时，就执行“乘法减小”算法，把慢开始门限减半，这是为了预防网络发生拥塞。
- 由于发送方现在认为网络很可能没有发生拥塞，因此现在不执行慢开始算法，而是把 cwnd 值设置为慢开始门限减半后的值，然后开始执行拥塞避免算法，是拥塞窗口的线性增大。

#### SYN攻击

在三次握手过程中，Server发送SYN-ACK之后，收到Client的ACK之前的TCP连接称为半连接（half-open connect），此时Server处于SYN_RCVD状态，当收到ACK后，Server转入ESTABLISHED状态。

SYN攻击就是 Client 在短时间内伪造大量不存在的 IP 地址，并向 Server 不断地发送 SYN 包，Server 回复确认包，并等待 Client 的确认，由于源地址是不存在的，因此，Server 需要不断重发直至超时。

这些伪造的SYN包将产时间占用未连接队列（内核会为每个这样的连接分配资源的），导致正常的SYN请求因为队列满而被丢弃，从而引起网络堵塞甚至系统瘫痪。

SYN攻击时一种典型的DDOS攻击，检测SYN攻击的方式非常简单，即当Server上有大量半连接状态且源IP地址是随机的，则可以断定遭到SYN攻击了

#### OSI七层模型介绍

- 7 层：应用层——为应用程序提供服务，提供了针对特定应用的协议（ftp, SMTP, Telnet远程登录，DNS， http等）
- 6 层：表示层——数据格式的转换（压缩、解压缩、加密解密等）
- 5 层：会话层——负责建立和断开通信连接（管理通信双方的会话，如何时建立连接，何时断开连接，保持多久连接等）
- 4 层：传输层——用于进程之间的数据传输。负责可靠传输
- 3 层：网络层——用于不同主机之间的数据传输（具体功能是IP分组的转发、路由选择等）
- 2 层：数据链路层——接收网络层的数据，封装形成数据帧，用于在物理网络上的节点之间传送数据！
- 1 层：物理层——提供实际的物理传输媒介



#### ARP地址解析协议

- 即 ARP（Address Resolution Protocol），是根据 IP 地址获取物理地址的一个TCP/IP协议。
- 主机发送信息时将包含目标 IP 地址的 ARP 请求广播到网络上的所有主机，并接收返回消息，以此确定目标的物理地址；收到返回消息后将该 IP 地址写入到本地的 ARP 缓存中。



#### TCP的连接与释放

1、首先 Client 端发送连接请求的报文（SYN=1）

2、Server 端接受了这个请求后，向客户端回复一个回复 ACK 报文（确认信号）；

3、Client 端接收到 ACK 报文后，也向Server 端发送一个确认报文，表明可以发送数据，这样一个TCP连接就建立了。

#### 为什么连接的时候是三次握手，关闭的时候却是四次握手？

因为当Server端收到Client端的SYN连接请求报文后，可以直接发送SYN+ACK报文。其中ACK报文是用来应答的，SYN报文是用来同步的。

但是关闭连接时，当Server端收到FIN报文时，很可能并不会立即关闭SOCKET，所以只能先回复一个ACK报文，告诉Client端，"你发的FIN报文我收到了"。

只有等到我Server端所有的报文都发送完了，我才能发送FIN报文，因此不能一起发送。故需要四步握手。

**TCP三次握手状态图**

![](H:\软件学习资料\学习笔记-2019\pictures\TCP三次握手.png)

**TCP四次挥手状态图**

![](H:\软件学习资料\学习笔记-2019\pictures\TCP四次挥手.png)

#### 四次挥手的原因

客户端发送了 FIN 连接释放报文之后，服务器收到了这个报文，就进入了 CLOSE-WAIT 状态。这个状态是为了让服务器端发送还未传送完毕的数据，传送完毕之后，服务器会发送 FIN 连接释放报文。



#### TIME_WAIT 的作用

客户端接收到服务器端的 FIN 报文后进入此状态，此时并不是直接进入 CLOSED 状态，还需要等待一个时间计时器设置的时间 2MSL。这么做有两个理由：

- 确保最后一个确认报文段能够到达。如果 B 没收到 A 发送来的确认报文段，那么就会重新发送连接释放请求报文段，A 等待一段时间就是为了处理这种情况的发生
- 等待一段时间是为了让本连接持续时间内所产生的所有报文段都从网络中消失，使得下一个新的连接不会出现旧的连接请求报文段。



### HTTP相关知识点

**get 与 post区别**

- 语义上，GET 是获取指定 URL 上的资源; 而 post向服务器传送数据，可以对指定的资源进行修改。

- Get 把请求的数据放在URL上，即 HTTP 协议头上，其格式为：以?分割 URL和请求数据，参数之间以 & 相连。 

- post把提交的数据放在 HTTP 包的内容实体中（requrest body）。

- 浏览器对URL的长度有限制，因此get提交的数据大小有限；而post方法提交的数据没有限制。

- get 请求参数会被完整保留在浏览历史记录里，而 post 中的参数不会被保留。

- GET 和 POST 本质上就是 TCP 连接，并无差别。但是由于 HTTP 的规定和浏览器/服务器的限制，导致他们在应用过程中体现出一些不同。

- GET 产生一个 TCP 数据包；POST 产生两个 TCP 数据包。

- 对于 GET 方式的请求，浏览器会把 http header 和 data 一并发送出去，服务器响应200（返回数据）；而对于 POST，浏览器先发送 header，服务器响应 100 continue，浏览器再发送 data，服务器响应200 ok（返回数据）。

**http 请求行：方法  URL  协议类型**

```http
GET /test/demo_form.asp?name1=value1&name2=value2 HTTP/1.1
```

```http
POST /test/demo_form.asp HTTP/1.1
Host: w3schools.com
name1=value1&name2=value2
```

**两者的应用场景：**

- GET：一般用于信息的获取，使用URL传递参数
- POST：一般用于修改服务器上的资源。



## 九 Linux I/O 模型

### select,  poll,  epoll 的对比

select、poll、epoll 都是 I/O 多路复用的机制。

I/O 多路复用机制，可以监视多个描述符，一旦某个描述符就绪（一般是读就绪或者写就绪），能够通知程序进行相应的读写操作。

但 select、poll、epoll本质上都是同步 I/O，因为它们都需要在读写事件就绪后自己负责进行读写，也就是说这个读写过程是阻塞的，而异步 I/O 则无需自己负责进行读写，异步 I/O 的实现会负责把数据从内核拷贝到用户空间。



I/O 多路复用，也称为 I/O 多路转接，APUE 上只有这样一句话。

为了使用这种技术，先构建一张我们感兴趣的描述符（通常都不止一个）的列表，然后调用一个函数，直到这些描述符中的一个已准备好进行 I/O 时，该函数才返回。

poll、pselect 和 select 这 3 个函数使我们能够执行 I/O 多路转接。在从这些函数返回时，进程会被告知哪些描述符已准备好可以进行 I/O。



#### Select 函数参数解析

```c++
int select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds, struct timeval *timeout);
```

  返回值：准备就绪的描述符数目：若超时，返回 0；若出错，返回 -1 
  参数 timeout，它指定愿意等待的时间长度，单位为秒和微妙。
（NULL：永远等待，当捕捉到已经准备好的描述符，则终止等待；
  Timeout != 0，等待指定的秒数和微妙数。当有描述符准备好，或者指定的时间已到，则立即返回；
  Timeout == 0, 根本不等待，测试所有的描述符并立即返回，即采用轮询的方式检测描述符，而不阻塞！
  select 第一个参数 nfds 的意思是“最大文件描述符编号值加 1”
  考虑所有 3 个描述符集，在 3 个描述符集中找出最大描述符编号值，然后加 1，这就是第一个参数值。
  也可将第一个参数设置为?FD_SETSIZE，这是 <sys/select.h> 中的一个常量，它指定最大描述符数（经常是 1024），但是对大多数应用程序而言，此值太大了。）
  select一次可以监测FD_SETSIZE数量大小的描述符，FD_SETSIZE通常是一个编译后指定的数量！

**select的返回值**

返回-1表示出错
返回 0 描述没有描述符准备好！
返回正值：表示已经准备好的描述符数量！该值是 3 个描述符集中已准备好的描述符数之和，所以如果同一描述符已准备好读和写，那么在返回值中会对其计两次数。
通过 select 返回值，做判断语句：负值，select错误；正值，某些文件可读、写或异常；0，等待超时，没有可读、写或异常的文件。

#### poll 函数

poll函数类似于select, 但是函数结构不同！

```c++
#include <poll.h>
    int poll(struct pollfd *fds, nfds_t nfds, int timeout);

返回值: 准备就绪的描述符数目; 若超时, 返回 0； 若出错, 返回 -1.
fds 中的元素是由 nfds 指定。poll 一次可以监测的描述符数量没有限制！，但也需要通过轮询的方式检查哪些描述符就绪！
参数解析：
与 select 不同，poll 不是为每个条件（可读性、可写性和异常条件）构造一个描述符集，而是构造一个 pollfd 结构的数组，每个数组元素指定一个描述符编号以及对该描述符感兴趣的条件。
struct pollfd {
    int   fd;         /* file descriptor */
    short events;     /* requested events */
    short revents;    /* returned events */
};
其中的 events 可取值为：pollout, pollhup, pollin 等，通过这些值告诉内核我们关心的是描述的哪些哪些事件！
```

#### epoll函数

```c++
#include <sys/epoll.h>
int epoll_create(int size);
int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event);
int epoll_wait(int epfd, struct epoll_event *events, int maxevents, int timeout);
			
(1) epoll_create用来创建一个epoll的句柄（对象），size 是告诉内核这个监听的数量有多大！
当创建好 epoll 句柄后，它就是会占用一个 fd 值。
调用 create 函数时，内核会创建一个 eventpoll 结构体，里面存放添加进来的 fd。这些 fd 是挂载到红黑树中！
```

**epoll_ctl：事件注册函数**
```c++
epoll_ctl：事件注册函数。epfd为epoll_create创建的对象
op 表示操作类型 (
    EPOLL_CTL_ADD：注册新的fd到epfd中；
    EPOLL_CTL_MOD：修改已经注册的fd的监听事件；
    EPOLL_CTL_DEL：从epfd中删除一个fd；
)
fd 是要监听的文件描述符。（实际上每个fd都有一个对应的回调函数，当文件就绪时，调用回调函数把就绪 fd 加入到一个就绪队列中）
所有添加到 epoll 对象的事件都会与设备驱动程序建立回调关系，即当相应的事件发生时，调用这个回调方法
event 是告诉内核要监听什么事，struct epoll_event 结构如下
// 感兴趣的事件和被触发的事件 
struct epoll_event { 
    __uint32_t events; /* Epoll events */ 
    epoll_data_t data; /* User data variable */ 
}
其中的 events 可以是如下宏的集合：
EPOLLIN ：表示对应的文件描述符可以读（包括对端SOCKET正常关闭）；
EPOLLOUT：表示对应的文件描述符可以写；
EPOLLPRI：表示对应的文件描述符有紧急的数据可读（这里应该表示有带外数据到来）；
EPOLLERR：表示对应的文件描述符发生错误；
EPOLLHUP：表示对应的文件描述符被挂断；
EPOLLET： 将EPOLL设为边缘触发(Edge Triggered)模式，这是相对于水平触发(Level Triggered)来说的。
EPOLLONESHOT：只监听一次事件，当监听完这次事件之后，如果还需要继续监听这个socket的话，需要再次把这个 socket 加入到 EPOLL 队列里
```

**epoll_wait 函数**

```c++
(3) epoll_wait：在fd就绪队列中检查是否有事件就绪，类似于select, 函数返回需要处理的事件数目。
	工作模式：LT（level trigger）和 ET（edge trigger）。
	LT 模式是默认模式，LT 模式与 ET 模式的区别如下：
	--LT模式--：当 epoll_wait 检测到描述符事件发生并将此事件通知应用程序，应用程序可以不立即处理该事件。下次调用 epoll_wait 时，会再次响应应用程序并通知此事件。
	--ET模式--：当 epoll_wait 检测到描述符事件发生并将此事件通知应用程序，应用程序必须立即处理该事件。如果不处理，下次调用 epoll_wait 时，不会再次响应应用程序并通知此事件。
	ET模式在很大程度上减少了 epoll 事件被重复触发的次数，因此效率要比 LT 模式高。epoll 工作在 ET 模式的时候，必须使用非阻塞套接口，以避免由于一个文件句柄的阻塞读/阻塞写操作把处理多个文件描述符的任务饿死。
```



#### select， poll， epoll 三者的对比

设想一下如下场景：有 100万 个客户端同时与一个服务器进程保持着TCP连接。而每一时刻，通常只有几百上千个 TCP 连接是活跃的(事实上大部分场景都是这种情况)。如何实现这样的高并发？

1、每次调用select/poll，服务器进程都会把100万个连接告诉操作系统（把fd描述符集从用户态copy到内核），让内核去查询这些fd集中是否有事件发生（可读、可写等就绪事件），轮询后，再把fd描述符集复制到用户态，让服务器进程轮询检查以发生的事件！这一过程开销比较大，效率低！

2、epoll 的机制是：通过 create 创建一个 epoll 对象，放在内核中；通过 ctl 函数向该对象中添加100万个fd；调用 epoll_wait 检查就绪队列中是否有 fd 就绪。内核不需要遍历全部的连接，只需要判断就绪队列是否为空！		



## 十 牛客刷题思路总结

1、判断二叉树的子结构

输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

解题思路：

* B 是 A 的子结构，有以下几种情况：
* 1-B 与 A 的根结点相同（此时 B 的子树的根必然与 A 的子树的根结点相同）
* 2-B 是 A 的左子树的子结构，不考虑 A 的根节点
* 3-B 是 A 的右子树的子结构，不考虑 A 的根节点
* 因此针对 2 和 3，可递归处理！对于情况1，需要单独写一个函数进行判断



2、判断链表是否有环-leetcode141

解题思路：
* 快慢指针
* 若需要求出入口节点，也可以使用该方法！即找到相遇节点后，一个指针从头节点开始移动，另一个指针从相遇点开始移动，最终两者在环的入口处相遇。

```c++
class Solution {
public:
    bool hasCycle(ListNode *head) {
        if(head == nullptr || head->next == nullptr)
            return false;
        ListNode *fast = head;
        ListNode *slow = head;
        while (fast != nullptr && fast->next != nullptr) {
            fast = fast->next->next;
            slow = slow->next;
            if (slow == fast)
                return true;
        }
        // 如果while循环内没有执行return语句，说明链表没有环
        return false;
    }
};
```

3、句子翻转-leetcode151

解题思路：先整体翻转，然后针对句中的每个单词进行翻转。
时空复杂度要求 O(N) 和 O(1)

leetcode上刷题采用的几种方式：
```c++
// 方法一：整体翻转+局部翻转
void reverseWords(string &s) {
    if (s.size() == 0)
        return;
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
        else // s[j] != ' '，j 继续移动 
            ++j;
    }
    // 若句子最后有空格，则退出 while 循环时，start = j = len;
    // 若句子最后没有空格，则最后一个单词没有还没有实现翻转。
    reverse(s.begin()+start, s.begin()+j); 
    // 去除句子最右边的空格
    if (s.back() == ' ')
        s.pop_back();
}
```



方法二：利用 stringstream 提取字符串。从前向后遍历字符串，并插入到 res 的首部，即可实现单词的翻转

```C++
// 时空复杂度均为 O(n)
class Solution {
public:
    void reverseWords(string &s) {
        if (s.empty())
            return;
        
        stringstream ss(s); // 把s的内容写入到字符流中
        string res = "";
        string tmp;
        while (ss >> tmp) {
            res.insert(0, tmp + ' '); // 单词之间有空格
        }
        
        // 若s中只有空格，ss>>tmp会忽略空格字符，此时res为空
        if (res.size() > 0) 
            res.pop_back(); // 去掉句子最后的空格
        
        s = res;
    }
};
```

方法三：从左到右遍历s，拼接每个单词
```c++
	void reverseWords(string &s) {
        if (s.empty())
            return;
        
        string res = "", tmp = "";
        int i = 0, len = s.size();
        
        while (i < len) {
            if (s[i] != ' ') {
                tmp += s[i]; // 提取单词
                ++i;
            }
            else { // 遇到 ' ', 通过循环跳过多个空格，找到下一个单词的位置
                while (i < len && s[i] == ' ')
                    ++i;
                res.insert(0, tmp + ' '); // 类似链表的头插法
                tmp = ""; //开始下一个单词的提取
            }
        }
        
        if (tmp.size() > 0) 
            res.insert(0, tmp + ' '); //最后一个单词
        
        while (res.back() == ' ')
            res.pop_back(); // 去除最后的空格
        s = res;
    }
```

4、Longest Increasing Subsequence（数组中的最长递增子序列）

思路一：动态规划思想，时间复杂度 O(N^2)

- lis[i]：表示 nums[0~i] 中以 nums[i] 作为结尾的最长递增子序列的长度
- 设 nums[k] 对应子序列中 nums[i] 的前一个数字
- 则有状态转移方程：LIST[i] = max {LIS[i], LIS[k]+1 }, for k < i && nums[k] < nums[i] 

```c++
 for (int i = 1; i < len; i++) {
 	for (int k = 0; k < i; k++) {
    	if (nums[k] < nums[i])
        	LIS[i] = max(LIS[i], LIS[k]+1);
    }
 }
```

思路二：遍历数组，把元素依次插入到结果数组 res。

- 设当前遍历到的元素为：num[i]，若 num[i] 大于 res 中所有元素，则把 num[i] 添加 res 中；
- 否则，利用 lower_bound() 找到res中第一个 >=num[i] 的数字 \*it，用 num[i] 替换 \*it, 此时res[0]~num[i] 之间的数字保持为递增序列。
- 当遍历完所有的元素之后，res 中的元素个数即为 LCS 的长度

```c++
for (int i = 0; i < len; i++) {
	auto it = lower_bound(res.begin(), res.end(), nums[i]); // 在res中查找第一个 >= nums[i]的元素
		if (it == res.end()) // 没有找到，说明 res 中的元素都比nums[i]小
        	res.push_back(nums[i]);
        else // it 指向第一个>=nums[i]的元素，替换之
        	*it = nums[i];
}
```



5、Top K Frequent Elements（统计出现次数最多的前k个元素）

method 1：使用 hash 表统计次数，以次数作为 key 存入priority_queue中，然后直接提取后 k 个元素

method 2：桶排序。根据元素出现的个数将元素分配到若干桶中，比如a0出现m次，则将a0放入桶Bm中。这样，序号最大的桶存放的便是出现次数最多的元素



```c++
vector<int> topKFrequent(vector<int>& nums, int k) {
    unordered_map<int, int> m;
    vector<vector<int>> bucket(nums.size() + 1);
    vector<int> res;
    for (auto a : nums) //计数
    	++m[a];
    for (auto it : m) 
    	bucket[it.second].push_back(it.first); // 每个桶是以元素出现的次数作为下标
	for (int i = nums.size(); i >= 0; --i) {
		for (int j = 0; j < bucket[i].size(); ++j) {
			res.push_back(bucket[i][j]);
			if (res.size() == k) 
				return res;
			}
		}
	return res;
}
```

6、求解 pow(x, n)

递归：先求出 (x ^ n/2)

```c++
double myPow(double x, int n) {
	if (n == 0)
		return 1;
        //if (n == 1) 
            //return x;
    if (n < 0) 
        return 1.0/x * pow(1/x, -(n+1)); // 防止n取INT_MIN时，-n溢出
    if (n % 2 == 0)
        return pow(x*x, n/2);
    else 
        return pow(x*x, n/2)*x; 
}
```

7、判断是否为 2 的幂

bit运算。若n为2的m次幂，显然n的二进制中只有一个1，因此通过n&(n-1)消去一个1，之后其值为0！
n& (n-1) == 0 ?  Yes : no



#### subsets

回溯法，递归

回溯法的基本代码流程:

```c++
backTrack(res, cur, num, position) {
    if (condition) { //递归终止条件
        xxx;
        return;
    }
    
    for (int i = position; i < num.size(); i++) {
        xxx;
        backTrack(xx,xx,xx,xx);
        cur.pop_back(); // 回溯
    }
}
```

**leecode-subsets**

```c++
class Solution {
public:
    vector<vector<int> > subsets(vector<int> &S) {
       vector<vector<int>> res;
        if (S.empty())
            return res;
        vector<int> sub; // 用于存放每一个子集
        res.push_back(vector<int>()); // 空集合
        sort(S.begin(), S.end()); // 先排序保证初始几何是有序的
        
        subset(res, sub, S, 0);
       return res;
    }
    
    // 回溯法思想
    void subset(vector<vector<int>> &res, vector<int> &sub, vector<int> &S, int start)
    {
        if (start == S.size())  // 返回，即开始回溯
            //res.push_back(sub);
            return;
// 第1个循环得到的是s[0]为开头的所有子集，第二次循环得到的是以s[1]为开头的子集...依次类推
        for (int i = start; i < S.size(); i++) {
            //先确定第一个元素
            sub.push_back(S[i]);
            // 先把当前子集sub存到res中
            res.push_back(sub); 
          // 递归处理后面的元素，注意是处理 i 之后的元素，得到以 s[i] 开头的组合
            subset(res, sub, S, i + 1);

// 弹出当前子集sub中元素，用于回溯时存放下一个元素（即当前for循环的下一个值start+1）
            sub.pop_back(); 
        }
    }
};
```



#### 字典树的构造

Trie结点的定义：

```c++
// 字典树的简单实现 2018-10-5
// 每个节点有26个分支，分别表示字母 a~z, a对应的分支为next[0], 则其余字母x的分支next[x-'a']
class TrieNode {
public:
    TrieNode *next[26]; // 每个节点有26个分支，字母种类为26
    bool isWord; // 判断root到当前节点是否形成一个单词
    
    TrieNode(bool flag = false) : isWord(flag), next{NULL}
    {
        //memset(next, 0, sizeof(next)); //next数组共26*4个字节
    }
};
```
字典树功能类的定义：

```c++
class Trie {
private: 
    TrieNode *root;
public:
    /** Initialize your data structure here. */
    Trie() : root(new TrieNode()) {}
    
    /** Inserts a word into the trie. */
    void insert(string word) {
        TrieNode *p = root;
        // 依次确定word中每个字母的位置
        for (int i = 0; i < word.size(); i++) {
            // str[i]对应的分支不存在，则创建相应的节点
            if (p->next[word[i] - 'a'] == NULL) // ->和[]优先级相同，这里相当于(p->next)[i]
                p->next[word[i] - 'a'] = new TrieNode();
            
            p = p->next[word[i] - 'a'];
        }
        // 当前word中的字母全部插入后，从root 到 p 节点的路径上已经形成了一个单词
        p->isWord = true;
    }
    
    /** 判断一个单词时否在该字典树中 */
    bool search(string word) {
        TrieNode *p = root; // 从root节点开始,根据word中的字母确定分支，逐步向下查找，直到找到单词的最后一个字母。
        int len = word.size();
        for (int i = 0; i < len && p != NULL; i++) { // p == null时，表明当前字母不存在分支
            p = p->next[word[i] - 'a']; // p移动到当前字母对应分支下的子节点
        }        
        return p != NULL && p->isWord;
    }
    
    // 判断字典树中是否有单词以prefix为前缀
    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix) {
        // 本题跟search类似，但是只要匹配前缀即可，不需要是一个完整的单词，所以不用考虑isWord。
        TrieNode *p = root;
        int len = prefix.size();
        for (int i = 0; i < len && p != NULL; i++) { 
            p = p->next[prefix[i] - 'a']; 
        }
        return p != NULL;
    }
};
```