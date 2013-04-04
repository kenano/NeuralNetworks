


function [calculated_output] = forward_pass( x_input , w , b )

required_output = tan(sort(x_input));
calculated_output = [];

count1 =1;

w_input = w(1);
w_hidden = [w(1,2) w(1,3) w(1,4)];
w_output = [w(1,5) w(1,6) w(1,7)];

b_input = b(1);
b_hidden = [b(1,2) b(1,3) b(1,4)];
b_output = [b(1,5)];

for i = sort(x_input)
	
	x_hidden = [ 1 1 1 ];
	x_output = [ 1 1 1 ];


	y_first = [ 1 ];
	y_second = [ 1 1 1 ];
	y_third = [ 1 ];
	
	% layer 1
	y_first = i * w_input(1) + b_input(1);
	out = sigmoidd(y_first);
	x_hidden = x_hidden * out;

	y_second = forward_hidden(x_hidden,w_hidden,b_hidden);

	% layer 2
	k = 1;	
	for j = y_second
		x_output(k) = sigmoidd(j);
		k = k + 1;	
	endfor;

	% layer 3
	y_third = forward_output(x_output, w_output, b_output);
	y_target = sigmoidd(y_third);

	calculated_output(count1) = y_target;
	count1 = count1 +1;
endfor;




end

%Sigmoid function G(s)
function [ out ] = sigmoidd( s )
	out = (1 - exp(-2*s)) ./ (1 + exp(-2*s));
end

function [ out ] = forward_hidden( x , w , b )
	j = 1;
	for i = x
		out(j)= w(j)*i + b(j);
		%disp(out(j));
		j= j + 1;
	endfor;
end

function [ out ] = forward_output( x , w , b )
	x_t = transpose(x);
	out = w*x_t;
	out = out + b;
end