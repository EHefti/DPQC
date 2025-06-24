function K = calculateEuclideanDist(P,Q)
% Vectorized method to compute pairwise Euclidean distance
% Returns K(i,j) = sqrt((P(i,:) - Q(j,:))'*(P(i,:) - Q(j,:)))

[nP, d] = size(P);
[nQ, d] = size(Q);

pmag = sum(P .* P, 2);
qmag = sum(Q .* Q, 2);

K = sqrt(ones(nP,1)*qmag' + pmag*ones(1,nQ) - 2*P*Q');

end