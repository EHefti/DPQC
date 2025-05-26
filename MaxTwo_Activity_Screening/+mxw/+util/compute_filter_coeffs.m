%% Compute and store locally bandpass coefficients for filtering without
%% any toolboxes

% compute some useful coefficients

order = 4:6;
hp=[5 10:10:100 100:50:800];
lp=[2000:50:8000];
sr=20000;

nyq = 0.5 * sr;

%%
clear stored_filters
for o = 1:length(order)
    clear b_all a_all
    for i=1:length(hp)
        for j=1:length(lp)
            
            low = hp(i) / nyq;
            high = lp(i) / nyq;
            [b a] = butter(order(o), [low high], 'bandpass');
            
            %         bp=fdesign.bandpass('n,f3dB1,f3dB2', 2, hp(i), lp(j), sr);
            %         %y.filters.bbp=butter(bp);
            %         bbp=design(bp,'butter','sosscalenorm','l1');
            %         [b, a] = sos2tf(bbp.sosMatrix, prod(bbp.ScaleValues));
            %
            b_all(:,i,j)=b;
            a_all(:,i,j)=a;
            
        end
    end
    stored_filters{o}.order = order(o);
    stored_filters{o}.hp=hp;
    stored_filters{o}.lp=lp;
    stored_filters{o}.b=b_all;
    stored_filters{o}.a=a_all;
    
end

%% store them


f= which('mxw.util.compute_filter_coeffs');
[pathstr, name, ext]=fileparts(f);

save([pathstr '/stored_filter_coeffs.mat'], 'stored_filters')


