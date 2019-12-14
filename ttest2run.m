
display_stats(rbfClassification,polyClassification);
display_stats(rbfClassification,linearClassification);
display_stats(rbfClassification,DT);
display_stats(linearClassification,DT);
display_stats(linearClassification,polyClassification);
display_stats(DT,polyClassification);

function display_stats(compare1, compare2)
   [h,p,ci,stats] = ttest2(compare1,compare2);
   disp('------------------------');
   disp(h);
   disp(p);
   disp(ci);
   disp(stats);
   disp('------------------------');
end
