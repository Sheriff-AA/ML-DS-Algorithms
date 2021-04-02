load data
new_cv = [2, 2; 3, 2];

D = [1, 2; 2, 1; 3, 1; 5, 5; 5, 6; 6, 5; 6, 6];
% C = [0.4, 0.7; 0.6, 0.3];

figure 
scatter(D(1:end, 1), D(1:end, 2));
hold on
scatter(C(1:end, 1), C(1:end, 2));
axis([0 8 0 8])
grid on
n = 0;

while n < 10
    new_cv = competitive_learning(D, new_cv, 0.4);

    figure 
    scatter(D(1:end, 1), D(1:end, 2));
    hold on
    scatter(new_cv(1:end, 1), new_cv(1:end, 2));
    axis([0 8 0 8])
    grid on
    n = n+1;
end