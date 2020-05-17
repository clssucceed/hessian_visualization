# hessian_visualization

通过人工设计一些合理的运动，特征分布来看下Hessian矩阵，通常会以热力图的方式查看Hessian矩阵

TODO:
1. 纯视觉 full-BA
2. 纯imu full-BA
3. 纯视觉 sliding window
4. 纯imu sliding window
5. vio full-BA
6. vio sliding window
7. 有先验/无先验

目标:
1. 从Hessian矩阵的角度理解drift和可观的区别
2. 直观理解历史观测如何通过marginalization factor对于滑窗中的状态产生影响的

