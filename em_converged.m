function converged = em_converged(loglik, previous_loglik, threshold)
if nargin < 3
threshold = 1e-4;
end
%log likelihood should increase
if loglik - previous_loglik < -1e-3 
fprintf(1,'******likelihood decreased from %6.4f to %6.4f!\n', previous_loglik,loglik);
end
delta_loglik = abs(loglik - previous_loglik);
avg_loglik = (abs(loglik) + abs(previous_loglik) + eps)/2;
if (delta_loglik / avg_loglik) < threshold
converged = 1;
else converged = 0; 
end
end