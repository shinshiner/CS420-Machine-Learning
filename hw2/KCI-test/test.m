% new1 1. CRIM: per capita crime rate by town 
% new2 2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft. 
% new3 3. INDUS: proportion of non-retail business acres per town 
% new4 5. NOX: nitric oxides concentration (parts per 10 million) 
% new5 6. RM: average number of rooms per dwelling 
% new6 7. AGE: proportion of owner-occupied units built prior to 1940 
% new7 8. DIS: weighted distances to five Boston employment centres 
% new8 10. TAX: full-value property-tax rate per $10,000 
% new9 11. PTRATIO: pupil-teacher ratio by town 
% new10 12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town 
% new11 13. LSTAT: % lower status of the population 
% new12 14. MEDV: Median value of owner-occupied homes in $1000's


data = load('d:\Documents\ml2\hw2\KCI-test\data\boston_wout_discrete.dat');
% x = data(:,11);
% y = data(:,5);
% z = data(:,12);
% p_val = UInd_KCItest(x,y);
% disp(p_val);

% [p_val, stat]=indtest_new(x,z,[],[])
% [p_val, stat]=indtest_new(x,z,y,[])
% disp(p_val);
f = fopen('d:\Documents\ml2\hw2\KCI-test\res.txt', 'w');
z = data(:,8);
num = 0;
err = 0;

tic;
try
    for i=1:12
        x = data(:,i);
        for j=1:12
            y = data(:,j);
            [p_val, stat] = indtest_new(x,y,z,[]);
            if p_val <= 0.001
    %                 disp(i);
    %                 disp(j);
    %                 disp(k);
    %                 disp(p_val);
                  num = num + 1;
            end
        end
    end
catch
    err = err + 1;
end
toc;

fprintf(f, '%d', num);
fprintf(f, '\r\n');
disp(num);
disp(err);
fclose(f);