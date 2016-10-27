clear all; close all; clc;
%%% Jiafeng Liao jxl155830
%%%%%%%% train data 
data = load('spam_train.data','r');
[length width] = size(data);
dimension = width - 1;
x = data(:,1:dimension);
y = data(:,width);
y(find(y==0)) = -1;
K = zeros([length]);
%%%%%%%%%%% validation data 
data_vali = load('spam_validation.data','r');
x_vali = data_vali(:,1:dimension);
y_vali = data_vali(:,width);
y_vali(find(y_vali==0)) = -1;
len_vali = size(y_vali,1);
K_vali = zeros([len_vali,length]);
%%%%%%%%%%%% test data
data_test = load('spam_test.data','r');
x_test = data_test(:,1:dimension);
y_test = data_test(:,width);
y_test(find(y_test==0)) = -1;
len_test = size(y_test,1);
K_test = zeros([len_test,length]);
%%%%%%%%%%%%%%%%%%%
sigma = [0.001 0.01 0.1 1 10 100];
c = [1 10 100 1000 10000];
% sigma = [100];
% c = [10000];
accuracy_train = zeros([size(sigma,2),size(c,2)]);
accuracy_vali = zeros([size(sigma,2),size(c,2)]);
accuracy_test = zeros([size(sigma,2),size(c,2)]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
for n = 1:1:size(sigma,2),
    %%%%%%%% train kernel, validation kernel and test kernel
    for i = 1:1:length,
        for j = 1:1:length,
            K(i,j) = exp(-norm(x(i,:)-x(j,:))^2/(2*sigma(n)^2));
        end
    end
    for i = 1:1:len_vali,
        for j = 1:1:length,
            K_vali(i,j) = exp(-norm(x_vali(i,:)-x(j,:))^2/(2*sigma(n)^2));
        end
    end
    for i = 1:1:len_test,
        for j = 1:1:length,
            K_test(i,j) = exp(-norm(x_test(i,:)-x(j,:))^2/(2*sigma(n)^2));
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for m = 1:1:size(c,2),
        H = repmat(y,[1,length]).*repmat(y',[length,1]).*K;
        % min 0.5*x'*H*x + f'*x   subject to:  A*x <= b Aeq*x = beq
        f = -1 * ones([length,1]); A = []; b = []; 
        Aeq = y'; beq = 0; LB = zeros([length,1]); UB = c(m) * ones([length,1]);
        [lambda,FVAL,EXITFLAG,OUTPUT,LAMBDA] = quadprog(H,f,A,b,Aeq,beq,LB,UB);
        %%%%%%%%%%%%% concept %%%%%%%%%%%%%%%%%%
        % weight * phi(x) = sum (lambda * y * k(x,xi))
        % bias = y - sum(lambda * y * k(x,xi))
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        lambda_filter = round(lambda*10)/10;  %%%% remove precision error
        temp1 = find(0<lambda_filter);          %%% filter lambda
        temp2 = find(lambda_filter < c(m));
        location = intersect(temp1,temp2);
        bias = mean(y(location)-K(location,:)*(lambda.*y));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % prediction below   train/validation/test
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        w_phiX = K * (lambda.*y);
        predict_Y = w_phiX + bias;  
        test = -y.*predict_Y;
        accuracy_train(n,m)=size(find(test<=0),1)/size(test,1)*100
        
        w_phiX = K_vali * (lambda.*y);
        predict_Y = w_phiX + bias;  
        test = -y_vali.*predict_Y;
        accuracy_vali(n,m)=size(find(test<=0),1)/size(test,1)*100
        
        w_phiX = K_test * (lambda.*y);
        predict_Y = w_phiX + bias;  
        test = -y_test.*predict_Y;
        accuracy_test(n,m)=size(find(test<=0),1)/size(test,1)*100
        
    end
end
toc        