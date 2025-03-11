% Written by Kristina Pestaria Sinaga (kristinasinaga57@yahoo.co.id)

clear all;close all;clc
tic;
load MSRC.mat

points = X;
label  = Y;

points_view =size(points,2);
  %                 %TTTTTTTTTTTTTTTTTTTTTTTTTT
cluster_n =numel(unique(label));
points_n  =size(X{1},1);
m = 2;
eta = 1;
%%-----add  
N= points_n;
K = length(X);%%    k should equal points_view
sX = [cluster_n, N, K];


ModCount = 2;    % unfold_mode_number
lambda= 0;
V = length(X);
for qq=1:K
H{qq} = zeros(cluster_n,N/cluster_n);
W{qq} = zeros(cluster_n,N);
end

% WTT = cell(1,ModCount);
% for jj=1:ModCount 
%     WTT{jj} = W{jj};
% end

W_tensor = cat(3, W{:,:}); 
for v=1:ModCount
    para_ten{v} = lambda;
end

for jj=1:ModCount
    WT{jj} = W_tensor;
end

H_tensor = cat(3, H{:,:});
%%%%----------%

%% Normalization
for h = 1:points_view
    points{1,h} = normalize_data(points{1,h});
end


%% Cluster Centers initialization 
rng(1) 
for h=1:points_view
    [n(h),d(h)]=size(points{h});
    clust_cen{h} = rand(cluster_n,d(h));
end

%% MEMBERSHIP INITIALIZATION  -----------U initialization
  
for h=1:points_view
    [n(h),d(h)]=size(points{h});
    u{h}=rand(cluster_n,n(h));
    
    u_sum{h}=sum(u{h});
    u{h}=u{h}./repmat(u_sum{h},cluster_n,1);
    
end 
u0=u;

%% VIEW WEIGHT INITIALIZATION   ----------v initialization
rng(13)
x=rand([1 points_view]);
wv=x/sum(x,2);

t_max=70;
index_wf_red_all{h}=[];
SSIGMA = 0;


for itr =1:t_max
    
    alpha  = itr/points_n;
    beta   = itr/points_view;
    

    %% updating feature weights -------- update w
    
    for h=1:points_view
        temp9=0;
        
        for qq=1:cluster_n
            W1=bsxfun(@minus,points{h},clust_cen{h}(qq,:)).^2;
            for jj=1:points_n
            W2=bsxfun(@times,u{h}(qq,jj).^m,W1);
            
            for hh=1:points_view
                if hh~=h 
                    temp9 = temp9+alpha* (u{h}(qq,jj) - u{hh}(qq,jj))^m*W1;
                end
            end
            
            W3=sum(W2,1);
            temp10= sum(temp9,1);
            end
        
        end
        
        W4=(W3+temp10);
        W5=sum(W4,2);
        new_wf{h}=bsxfun(@rdivide,W4,W5); 
    end  
    
    
    %% DISCARD WEIGHT   

    for h=1:points_view
        
        [n(h),d(h)]=size(points{h});
        thres_reduce=1/sqrt(n(h)*d(h));   
        index_W_red{h}=find(new_wf{h}<thres_reduce);
        
        %adjusting features weight of h-th view
        new_wf{h}(index_W_red{h})=[];
        new_wf{h}=new_wf{h}/sum(new_wf{h});
        
        %adjusting points
        new_points{h}=points{h};
        new_points{h}(:,index_W_red{h})=[];
        
        %adjusting cluster center
        new_clust_cen{h}=clust_cen{h};
        new_clust_cen{h}(:,index_W_red{h})=[];
        index_wf_red_all{h}=[index_wf_red_all{h} index_W_red{h}];
        
    end
    
    points=new_points;
    wf=new_wf;
    clust_cen=new_clust_cen;
    


    %% updating view weights  -----------update  v
           
    V5=[];
    for h=1:points_view
        temp12=0;
        temp11=0;
        for qq=1:cluster_n
                
            for jj=1:points_n
                V1=new_wf{h}.^2.*GetDistance(points{h}(jj,:),clust_cen{h}(qq,:)).^2; %new_wf{h}.^2*GetDistance(points{h}(i,:),clust_cen{h}(k,:)).^2;%bsxfun(@minus,points_temp,clust_cen_temp(k,:));
                temp12=temp12+(u{h}(qq,jj).^m.*V1);
                for hh=1:points_view
                    if hh~=h
                       temp11 = temp11+alpha* (u{h}(qq,jj) - u{hh}(qq,jj))^m*V1;
                    end
                end
            end
            V3=temp12+temp11;
            
        end
        V4=sum(V3);
        V5=[V5 V4];

    end
    V6=V5.^(1/(beta-1));
    V7=sum(V6,2);
    new_wv=bsxfun(@rdivide,V6,V7);
    
    
    %% Updating cluster centers    ----------update a
    
    
    for h = 1:points_view
        for qq = 1:cluster_n 
            for s = 1:size(points{h},2)
            temp1 = 0; 
            temp2 = 0; 
            temp3 = 0; 
            temp4 = 0; 
            for jj = 1:points_n 
                temp1 = temp1 + new_wf{h}.^2*u{h}(qq,jj)^2*points{h}(jj,s); 
                temp2 = temp2 + new_wf{h}.^2*u{h}(qq,jj)^2; 
                for hh = 1:points_view 
                    if hh ~= h 
                       temp3 = temp3 + alpha*new_wf{h}.^2*(u{h}(qq,jj) - u{hh}(qq,jj))^2*points{h}(jj,s); 
                       temp4 = temp4 + alpha*new_wf{h}.^2*(u{h}(qq,jj) - u{hh}(qq,jj))^2; 
                    end 
                end 
            end 
            new_clust_cen{h}(qq,s) = (temp1 + temp3)/(temp2 + temp4); 
            end
                    
             
        end
            
    end

    
    %% Updating memberships  --------- update  U
    s1 = cell(1,V);
    
    for h=1:points_view   %        views  
        for jj=1:points_n  %%%      numbers of sample
            for qq=1:cluster_n  %%  number of class
                
                
                temp1=0;   % ----------------------- -the  third  item 
                temp2=0;                
                for hh=1:points_view
                    if hh ~= h 
                        temp1 = temp1 + alpha*u{hh}(qq,jj)*(new_wf{h}.^2*(GetDistance(points{h}(jj,:),new_clust_cen{h}(qq,:)).^2));  
                        temp2 = temp2 + alpha; 
                    end                   
                end
                
                
                trfrac=0.05:0.05:0.95;
                u_tensor = cat(3, u{:,:}); 
                nn=prod(size(u_tensor)); 
                
                for ii=1:length(trfrac)                                             
                    ntr=round(nn*trfrac(ii));
                    ind=randperm(nn); 
                    ind=ind(1:ntr)';
                    [I,J,K]=ind2sub(size(u_tensor),ind);
                    [S_tensor,H_tensor]=tensorconst_U(zeros(size(u_tensor)),{I,J,K},u_tensor(ind),0,1);  %%lambda = 0 eta = 1               
                end
                
                
                for hhh=1:V
                    s1{hhh} = S_tensor(:,:,hhh);
                end
                temptu = s1{hh}(qq,jj);
                
                  %u_tensor = cat(3, X{:,:});
%                 for hhh = 1:points_view
%                 temptu = 0;
%                 X1 = zeros(size(X{hhh}));                   
%                 for jjj =1:ModCount         %%%zhan kai K ci          
%                     X1 = X1 - flatten_adj(eta*WT{jjj},sX,jjj);
%                 end
%                 S=X1/(eta*ModCount);
%                 Sv = cell(1,points_view); 
%                 Sv{hhh}=S(:,:,hhh);
%                 temptu = temptu+ Sv{hhh};
%                 end
                
               tempdd = temp2*(new_wf{h}.^2*(GetDistance(points{h}(jj,:),new_clust_cen{h}(qq,:)).^2))+ ModCount*eta;                           
               tempthird = (temptu + temp1)/tempdd;
                
                        % ------------------------the second item           
                temp3 = 0; 
      
                for l = 1:cluster_n 
                    temp3 = temp3 + ((new_wf{h}.^2*(GetDistance(points{h}(jj,:),new_clust_cen{h}(qq,:)).^2)) + ModCount*eta )/  ((new_wf{h}.^2 * (GetDistance(points{h}(jj,:),new_clust_cen{h}(l,:)).^2))+ ModCount*eta); 
                end
                
                
       %----------------------------------------------------------------------------------------------------------
                %temp4 = 0;  %  -------------------------the first item   U  X      Z   H 
                for l = 1:cluster_n   %% l ---class   hh ---views
                    
                    temp = 0;                                   %%----1
                    for hh = 1:points_view 
                        if hh ~= h 
                            temp = temp + (alpha*u{hh}(l,jj)*(new_wf{h}.^2*(GetDistance(points{h}(jj,:),new_clust_cen{h}(l,:)).^2)));
                        %%%%
                         
                        end
                    end
         
                                                     % ----2
                for ii=1:length(trfrac)                                             
                    ntr=round(nn*trfrac(ii));
                    ind=randperm(nn); 
                    ind=ind(1:ntr)';
                    [I,J,K]=ind2sub(size(u_tensor),ind);
                    [S_tensor,H_tensor]=tensorconst_U(zeros(size(u_tensor)),{I,J,K},u_tensor(ind),0,1);  %%lambda = 0 eta = 1               
                end
                
                
                for dd=1:V
                    s1{dd} = S_tensor(:,:,dd);
                end              
                tempt = s1{hh}(l,jj);
                
                %----------------33333
               
                tempd = temp2*(new_wf{h}.^2*(GetDistance(points{h}(jj,:),new_clust_cen{h}(l,:)).^2))+ ModCount*eta; 
                              
                end
                             
                tempfirst = 1-(temp +tempt)/tempd;   
                
                new_u{h}(qq,jj) = tempfirst + temp3 + tempthird;   % temp1/(1 + temp2) +
                
            end
        end

    end    
    
    
    u=new_u;
    clust_cen=new_clust_cen;
    wf=new_wf;
    wv=new_wv;
    
    
        for h=1:points_view
            cost_temp(h)=sum(sum(((u0{h}-u{h}).^2)))/(points_n*cluster_n*points_view);  
        end
        cost(itr)=sum(cost_temp);
        fprintf('rate = %d, fitness = %f\n', itr, cost(itr))
        if itr>1
            if abs(cost(itr)-cost(itr-1))<0.00001 
                break;
            end
        end
 %% 4---------------------------------------------------------update G
    %V = length(X);
    mu = 10e-5; 
    
%     %twist-version
%    [g, objV] = wshrinkObj(z + 1/rho*w,1/rho,sX,0,3) ;
%     G_tensor = reshape(g, sX);  
%     %5 update W
%     w = w + rho*(z - g);
% 
     for umod=1:ModCount
%         H_tensor = updateG_tensor(WT{umod},u,sX,mu,para_ten,V,umod);  %para_ten--thata
%         %mu--eta constrain   
%         WT{umod} = WT{umod}+mu*(u_tensor-Z_tensor);
     end
   
%     for qq=1:3
%         H{qq} = H_tensor(:,:,qq);
% %        W_tensor = cat(3, WT{:,:});
%  %       W{qq} = W_tensor(:,:,qq);
%     end
    
   % fprintf('iter=%g', iter); 
    
end

%% Global Solution
uu=(u{1}.*wv(1))+(u{2}.*wv(2))+(u{3}.*wv(3))+(u{4}.*wv(4))+(u{5}.*wv(5));
    
    
clust=[];
for jj=1:points_n
    [num idx]=max(uu(:,jj));
    clust=[clust;idx];
end    

%% Evaluation Metrics
AR=1-ErrorRate(label,clust,cluster_n)/points_n;
[AR valid_external(label,clust) nmi(label,clust)]
toc;
    

