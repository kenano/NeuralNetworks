


x_input = rand(1,1);
x_hidden = [ 1 1 1 ];
x_output = [ 1 1 1 ];


y_first = [ 1 1 1 ];
y_second = [ 1 1 1 ];
y_third = [ 1 ];

w_input = rand(1,1);
w_hidden = rand(1,3);
w_output = rand(1,3);

b_input = rand(1,1);
b_hidden = rand(1,3);
b_output = rand(1,1);

delta_output = [ 1 ];
delta_hidden = [ 1 1 1 ];
delta_input = [ 1 ];


function [ out ] = sigmoidd( s )
	out = (1 - exp(-2*s)) ./ (1 + exp(-2*s));
end

function [ out ] = forward_hidden( x , w , b )
	j = 1;
	for i = x
		out(j)= w(j)*i + b(j);
		disp(out(j));
		j= j + 1;

	endfor; 
end

function [ out ] = forward_output( x , w , b )
	x_t = transpose(x);
	out = w*x_t;
	out = out + b;
end

function [ delta_hidden ] = propogate_delta_to_hidden(w_output, delta_output)
	j = 1;	
	for i = w_output
		delta_hidden(j) = i * delta_output;
		j = j+1;
	endfor;
end

function [ delta_input ] = propogate_delta_to_input(w_hidden, delta_hidden)
	w_hidden_t = transpose(w_hidden);
	delta_input = delta_hidden*w_hidden_t;
end



% Looping through all the input cases
for i = x_input 
%
% Forward Pass begins:
%
	disp(i) 
% layer 1 
	y_first = i * w_input(1) + b_input(1);     
	out = sigmoidd(y_first);  	
	x_hidden = x_hidden * out;
	
	y_second = [ 1 1 1 ];
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
	disp(y_target);

% find actual tan answer
	y_expected = tan(i);	
	

% calculating error
	delta_output = y_expected - y_target;

	delta_hidden = propogate_delta_to_hidden(w_output,delta_output);
	delta_input = propogate_delta_to_input(w_hidden,delta_hidden);


		


endfor;









