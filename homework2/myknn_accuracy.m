function [accuracy] =myknn_accuracy(X_train,Y_train,X_test,Y_test,k)
[row_train col_train] = size(X_train);
[row_test col_test] = size(X_test);

count = 0;
for i=1:row_test
    class = myknn(X_train,Y_train,X_test(i,:),k);
    if class == Y_test(i,1)
        count = count+1;
    end
end
accuracy = (count * 100)/ row_test;

end