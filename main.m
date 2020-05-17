close all;
clear;
clc;

%% 参数设置
frame_number = 10;
image_width = 1280;
image_height = 720;
grid_col = 16;
grid_row = 9;
grid_width = image_width / grid_col;
grid_height = image_height / grid_row;
feature_number_per_frame = grid_row * grid_col;
K = [1000, 0, 640; 0, 1000, 360; 0, 0, 1];
Kinv = inv(K);
focal_length = K(1, 1);

%% 初始化Hessian
H_size = 6 * frame_number + feature_number_per_frame * (frame_number - 1);
H = zeros(H_size, H_size);

%% 构建Hessian
Rwk = eye(3); % k表示key frame
twk = [0; 0; 0];
for index_frame = 1 : frame_number - 1
    fprintf("Frame %d\n", index_frame);
    %% 生成Twci
    % 沿着相机z轴匀速直线运动（１m/s）
    Rwc = eye(3); % c表示current frame
    twc = [0; 0; index_frame];
    %% 生成第i帧的视觉特征
    % 均匀分布(每个grid中随机生成一个特征，特征深度随机生成，每个特征只被相邻两帧看到)
    for index_grid_row = 1 : 9
        for index_grid_col = 1 : 16
            grid_row_begin = (index_grid_row - 1) * grid_height;
            grid_col_begin = (index_grid_col - 1) * grid_width;
            %% 在key frame上随机生成特征
            uk = grid_col_begin + rand * grid_width;
            vk = grid_row_begin + rand * grid_height;
            Zk = rand * 90 + 10; % 深度在10~100之间均匀分布
            %% 计算这个特征相关联的Jacobian（由于每个特征只会被相邻帧看到，所以只有一个相关联的residual）
            Pk = Zk * Kinv * [uk; vk; 1];
            Pw = Rwk * Pk + twk;
            Pc = Rwc' * (Pw - twc);
            dpc2_dPc = focal_length * ...
                [1 / Pc(3), 0, -Pc(1) / (Pc(3) * Pc(3)); ...
                 0, 1 / Pc(3), -Pc(2) / (Pc(3) * Pc(3))];
            dpc2_dPw = dpc2_dPc * Rwc';
            dr_dRwk = dpc2_dPw * (-Rwk * skew_matrix(Pk));
            dr_dtwk = dpc2_dPw;
            dr_dRwc = dpc2_dPc * skew_matrix(Pc);
            dr_dtwc = dpc2_dPc * (-Rwc');
            dr_ddk = dpc2_dPw * Rwk * (-Zk * Pk); % dk是key frame的逆深度
            J = [dr_dRwk, dr_dtwk, dr_dRwc, dr_dtwc, dr_ddk];
            %% 将这个视觉factor产生的Hessian矩阵累加到H
            H_key_frame_indexes = ((index_frame - 1) * 6 + 1) : (index_frame * 6);
            H_current_frame_indexes = (index_frame * 6 + 1) : (index_frame * 6 + 6);
            H_inverse_depth_indexes = frame_number * 6 + ...
                (index_frame - 1) * feature_number_per_frame + ...
                ((index_grid_row - 1) * grid_col + index_grid_col);
            H_indexes = [H_key_frame_indexes, H_current_frame_indexes, H_inverse_depth_indexes];
            H(H_indexes, H_indexes) = H(H_indexes, H_indexes) + J' * J;
        end
    end
    Rwk = Rwc;
    twk = twc;
end

%% 可视化Hessian
C = inv(H);
pose_indexes = 1 : frame_number * 6;
imshow(C(pose_indexes, pose_indexes));
colormap(jet);
colorbar;

%% 结论: 在没有noise和先验的情况下，所有pose的协防差没有明显的区别

function a_hat = skew_matrix(a)
a_hat = [0, -a(3), a(2); a(3), 0, -a(1); -a(2), a(1), 0];
end