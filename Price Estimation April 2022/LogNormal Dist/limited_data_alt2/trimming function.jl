# trimming function 

function trim(x,h, delta)
    if abs(x)>2*h^delta
        return 1 
    elseif abs(x)<h^delta
        return 0 
    else
        return ((4*(x-(h^delta))^3)/(h^(3*delta))) - ((3*(x-h^delta)^4)/(h^(4*delta)))
    end
    
end

trim(2.4e-7, 0.03, 5) 
plot(x-> trim(x, 0.4, 5), 0.4^5, 2*0.4^5)
0.03^5