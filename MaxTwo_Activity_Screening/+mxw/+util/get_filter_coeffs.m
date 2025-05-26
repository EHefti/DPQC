function [b a]=get_local_filter(order,hpf,lpf,sr)

%GET_LOCAL_BBP_FILTER(HPF,LPF,SR,OSF)
%   get prestored filters for using the hidens_software with the
%   measurement laptop locally
%
%   TYPE=1:  'hpf', 500, 'lpf', 3000
%
%   SR: sampling rate
%   OSF: oversampling factor

assert(sr==20000, 'locally stored filters only support sampling rate of 20kHz and no oversampled channels');

load('stored_filter_coeffs.mat');

for i=1:length(stored_filters)
    if stored_filters{i}.order == order
        o = i;
    end
end
assert(not(isempty(o)), 'No coefficients for this filter order stored.');

i_ind=find(stored_filters{o}.hp==hpf);
j_ind=find(stored_filters{o}.lp==lpf);
[~, cl_hp_i]=min(abs(stored_filters{o}.hp-hpf));
[~, cl_lp_i]=min(abs(stored_filters{o}.lp-lpf));

assert(not(isempty(i_ind)), 'could not load a local HP filter with matching cutoff.\Closest available cutoff is %d\n', stored_filters{o}.hp(cl_hp_i))
assert(not(isempty(j_ind)), 'could not load a local LP filter with matching cutoff.\nClosest available cutoff is %d\n', stored_filters{o}.lp(cl_lp_i))

b=stored_filters{o}.b(:,i_ind,j_ind);
a=stored_filters{o}.a(:,i_ind,j_ind);




