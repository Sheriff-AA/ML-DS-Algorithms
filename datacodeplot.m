function datacodeplot(data, codevector)
options = foptions();
options(14) = 1;
itr = 1;

while itr == 1
    figure
    scatter(data(1:end, 1), data(1:end, 2));
    hold on
    scatter(codevector(1:end, 1), codevector(1:end, 2));
    axis([0 8 0 8])
    grid on
    
    ax = codevector;
    [codevector, b, c, d] = kmeans(codevector, data, options);

    if codevector == ax
        itr = 0;
    end

end