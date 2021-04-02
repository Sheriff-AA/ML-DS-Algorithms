function updated_cv = competitive_learning(data, codevectors, learning_rate)
data_size = size(data);
rand_row = rand;
rand_row = floor(data_size(1) * rand_row);

if rand_row <= 0
    rand_row = 1;
end
    
row = size(codevectors);
row = row(1);
ax = sum(max(data).^2);
for index = 1:row
    distance = ((data(rand_row, 1:end)) - (codevectors(index, 1:end))).^2;
    distance = sum(distance);
    
    if distance < ax
        ax = distance;
        winning_cv = codevectors(index, 1:end);
        winning_index = index;
    end
end

new_cv = winning_cv + (learning_rate * (data(rand_row, 1:end) - winning_cv));
codevectors(winning_index, 1:end) = new_cv;
updated_cv = codevectors;
