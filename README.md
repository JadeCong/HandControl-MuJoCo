# Drive MuJoCo Mechanical Hand(MPL/Adriot)

> Teleoperate the virtual mechanical hand in MuJoCo using keypoint mapping based on live depth stream of user’s hand.

---

## 1. Description

> 1. Input: 标记好的人手关节点数据（21个关节点数据，见示意图片）
> 2. Process: 基于人手模型的关节点数据，通过传统的人手模型关节之间的位姿关系找出人手各关节点的之间的对应的相互联系，从而编写传统算法，解算出对应的人手模型的关节数据。（20个关节，每根手指四个关节，三个平面转动关节+1个根部摆动关节）
> 3. Output: 人手模型的在某一姿态下对应的20个关节角度数据

---

## 2. File Type(programming language/file)

> 1. Jupyter Notebook file(for test algorithm and show the result)
> 2. Python file(for mujoco mechanical hand joint pose calculation)
> 3. Xml file(for model definition)

---

## 3. Project Path & Version Info

> 1. Path: server01: /home/jade/DRL/codes/MuJoCo/
> 2. Version: 
>> 最好生成一个diff file, 每个diff file 包含对每个版本的改动: diff [-abBcdefHilnNpPqrstTuvwy][-<行数>][-C <行数>][-D <巨集名称>][-I <字符或字符串>][-S <文件>][-W <宽度>][-x <文件或目录>][-X <文件>][--help][--left-column][--suppress-common-line][文件或目录1][文件或目录2]
>>
>> Version1.0: 能够根据标注的人手关节点数据，初步结算出人手的关节角数据，从而驱动mujoco中机械手运动，并显示出机械手的三维模型。
>> Version2.0：（1）重新改写了算法，提高了准确度，目前达到本人认为比较好的水平；（2）重新修改了MPL机械手模型，从而使机械手的大拇指构型及运动更像人手；（3）增添了人手关键点的动态交互显示；（4）增加了mujoco机械手自碰撞干涉检测；（5）增加了制作数据集的功能，能从不同角度获取机械手的rgb和depth图；（6）增加了人手到机械手的关键点映射，从而输出机械手对应动作的关键点。

---

## 4. Dependency

> 1. Package：
>> (1) first_try: 用正则来匹配查找路径下的文件，生成匹配文件的列表<br>
>> (2) opencv: 用于处理并显示图片<br>
>> (3) pyplot: 用于显示图片<br>
>> (4) mujoco_py: 用于导入mujoco模型，仿真驱动机械手，渲染模型及场景
>
> 2. Dataset: 
>> 标注的人手关键点数据集：server00: /opt/playground/msra_importer/<br>
>> 基于OpenPose实时识别的视频帧人手关节点数据集：server01: /home/jade/DRL/datasets/picture/video_output/

---

## 5. How We Do

> This file is the description of how to drive the mujoco mechanical hand based on the human hand annotated key points.

---
