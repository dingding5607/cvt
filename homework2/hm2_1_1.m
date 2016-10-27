clear all; close all; clc;
%%% Jiafeng Liao jxl155830
%%%%%%% train data part
data = load('spam_train.data','r');
[length width] = size(data);
dimension = width - 1;
x = data(:,1:dimension);
y = data(:,width);
y(find(y==0)) = -1;
len_weight = dimension;
len_xi = length;
c = [1,10,100,1000,10000];
accuracy = [];weight=[];bias=[];
% min 0.5*x'*H*x + f'*x   subject to:  A*x <= b 
% variable = [weight b xi]
for i = 1:1:size(c,2),
    H = diag([ones([len_weight,1]);0;zeros([len_xi,1])]);
    f = [zeros([len_weight,1]);0;c(i)*ones([len_xi,1])];
    A = [-repmat(y,1,dimension).*x,-y,-eye(length)];
    b = -ones([length,1]);
    LB = [-inf(dimension,1);-inf;zeros([length,1])];
    [X,FVAL,EXITFLAG,OUTPUT,LAMBDA] = quadprog(H,f,A,b,[],[],LB);
    weight = [weight X(1:dimension)];
    bias = [bias X(dimension+1)];
    xi = X(dimension+2:size(X,1));
    test = -y.*(weight(:,i)'*x'+bias(:,i))';
    accuracy = [accuracy size(find(test<=0),1)/size(test,1)*100];
end
plot(accuracy(:));
%%%%%%%%%%%% validation data and accuracy
validation_data = load('spam_validation.data','r');
validation_x = validation_data(:,1:dimension);
validation_y = validation_data(:,width);
validation_y(find(validation_y==0)) = -1;
validation_accuracy = [];
for j = 1:1:size(c,2),
   validation_test = -validation_y.*(weight(:,j)'*validation_x'+bias(:,j))';
   validation_accuracy = [validation_accuracy size(find(validation_test<=0),1)/size(validation_test,1)*100];
end

%%%%%%%%%% test data accuracy
test_data = load('spam_test.data','r');
test_x = test_data(:,1:dimension);
test_y = test_data(:,width);
test_y(find(test_y==0)) = -1;
test_accuracy = [];
for j = 1:1:size(c,2),
   test_test = -test_y.*(weight(:,j)'*test_x'+bias(:,j))';
   test_accuracy = [test_accuracy size(find(test_test<=0),1)/size(test_test,1)*100];
end

