indices = crossvalind('Kfold', size(x_aut,2), 10);

y_test_meas =[];
y_train_meas =[];
y_test_pred =[];
y_train_pred =[];

for i =1:10
    
     test = (indices == i); train = ~test;
     
     x_train = x_aut(train)';
     x_test = x_aut(test)';
    
     y_train = y_aut(train)';
     y_test = y_aut(test)';
    
     b1 = x_train\y_train;
     
     y_test_meas= vertcat(y_test_meas,y_test);
     y_train_meas= vertcat(y_train_meas,y_train);

     y_test_pred= vertcat(y_test_pred, b1*x_test);
     y_train_meas= vertcat(y_train_meas,b1*x_train);

end