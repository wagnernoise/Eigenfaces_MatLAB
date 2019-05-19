% number of images on the training set. Eigenvalues
M = 14;
%read and show images(jpg);
% S will store all the images
S=[]; 
figure(1);

%Chosen std and mean.
%It can be any number that it is close to the std and mean of most of the images.
um=100;
ustd=80;

for i=1:M
    str = strcat(int2str(i)); 
    str = strcat(str, '.jpg'); 
    eval('img=imread(str);');
    img = img(:,:,[1 1 1]);
    img = rgb2gray(img);
    img = imresize(img, [300,300]);
    subplot(ceil(sqrt(M)),ceil(sqrt(M)),i) 
    imshow(img)
    if i==3
    title('Course Intructors','fontsize',14) 
    end
    drawnow;
% save the dimensions of the image (irow, icold)
    [irow, icol]=size(img);
% creates a (N1*N2) x 1 matrix and add to S 
    temp=reshape(img',irow*icol,1);
%S will eventually be a (N1*N2) x M matrix.
    S=[S, temp];
end

%Normalize. Eigenfaces
for i=1:size(S,2) 
    temp=double(S(:,i)); 
    m=mean(temp);
    st=std(temp); 
    S(:,i)=(temp-m)*ustd/st+um;
end
%save and show normalized images
figure(2);

for i=1:M
    str=strcat('eigenimages', int2str(i),'.jpg'); 
    img=reshape(S(:,i),icol,irow); 
    img=img';
    eval('imwrite(img,str)');
    subplot(ceil(sqrt(M)),ceil(sqrt(M)),i) 
    imshow(img)
    drawnow;
    if i==3
    title('Normalized Images','fontsize',18) 
    end
end

%mean face;
%obtains the mean of each row of each image
m=mean(S,2);
%convert to unsigned 8-bit integer. Values range from 0 to 255
tmimg=uint8(m);
%takes the vector and creates a matrix
img=reshape(tmimg,icol,irow);
%matrix transpose
img=img';
figure(3);
imshow(img);
title('Mean Image','fontsize',18)
meanImg = img;

% show the difference from mean
figure(4); 
for i=1:M
    str=strcat(int2str(i),'.jpg'); 
    img=reshape(S(:,i),icol,irow); 
    img=img';
    img = img - meanImg; 
    subplot(ceil(sqrt(M)),ceil(sqrt(M)),i) 
    imshow(img)
    drawnow;
    if i==3
        title('Difference of Normalized Images from the Mean','fontsize',18)
    end
end

% Compute A matrix
dbx=[];
for i=1:M
    temp=double(S(:,i));
    dbx=[dbx temp];
end
A=dbx';

%Covariance matrix C=A'A, L=AA'
L=A*A';
% vv are the eigenvector for L
% dd are the eigenvalue for both L=dbx'*dbx and C=dbx*dbx'; 
[vv dd]=eig(L);
% Sort and eliminate those whose eigenvalue is zero
v=[];
d=[];
for i=1:size(vv,2)
    if(dd(i,i)>1e-4)
        v=[v vv(:,i)];
        d=[d dd(i,i)];
    end
end
%sort, will return an ascending sequence
[B, index]=sort(d); 
ind=zeros(size(index)); 
dtemp=zeros(size(index)); 
vtemp=zeros(size(v)); 
len=length(index);

for i=1:len 
    dtemp(i)=B(len+1-i); 
    ind(i)=len+1-index(i); 
    vtemp(:,ind(i))=v(:,i);
end
d=dtemp;
v=vtemp;
%Normalization of eigenvectors
for i=1:size(v,2) 
    kk=v(:,i); 
    temp=sqrt(sum(kk.^2)); 
    v(:,i)=v(:,i)./temp;
end
%Eigenvectors of C matrix
u=[];
for i=1:size(v,2)
    temp=sqrt(d(i));
    u=[u (dbx*v(:,i))./temp];
end
%Normalization of eigenvectors of the C matrix
for i=1:size(u,2) 
    kk=u(:,i);
    temp=sqrt(sum(kk.^2)); 
    u(:,i)=u(:,i)./temp;
end
% show eigenfaces;
EigenFaces = []; 
figure(5);
for i=1:size(u,2)
    img=reshape(u(:,i),icol,irow); 
    img=img';
    img=histeq(img,255);
    str = strcat('eigenimages', int2str(i)); str = strcat(str, '.jpg'); 
    eval('imwrite(img,str)');
    EigenFaces = [EigenFaces, img];
    subplot(ceil(sqrt(M)),ceil(sqrt(M)),i) 
    imshow(img)
    drawnow;
    if i==3
        title('Eigenfaces','fontsize',18) 
    end
end

% Find the weight of each face for each image in the training set. 
% omega will store this information for the training set.
omega = [];
for h=1:size(dbx,2)
    WW=[];
    for i=1:size(u,2)
        t = u(:,i)';
        WeightOfImage = dot(t,dbx(:,h)'); 
        WW = [WW; WeightOfImage];
    end
    omega = [omega, WW];
end

% InputImage is the new (unseen) image
% ensure that it is the same dimension image as the training set 
% We assume this new image is titled ?new_image.jpg? 
img=imread('CV_photo.jpg');
img = imresize(img, [300,300]);
InputImage = rgb2gray(img);
figure(5)
subplot(1,2,1)
imshow(InputImage); 
colormap('gray');
title('New image','fontsize',18)
InImage=reshape(double(InputImage),irow*icol,1);
temp=InImage;
me=mean(temp);
st=std(temp);
temp=(temp-me)*ustd/st+um;
NormImage = temp;
Difference = temp-m;
p = []; 
aa=size(u,2); 
for i = 1:aa
    pare = dot(NormImage,u(:,i));
    p = [p; pare];
end
%m is the mean image, u is the eigenvector
ReshapedImage = m + u(:,1:aa)*p;
ReshapedImage = reshape(ReshapedImage,irow,icol); 
ReshapedImage = ReshapedImage';
%show the reconstructed image.
subplot(1,2,2)
imagesc(ReshapedImage); 
colormap('gray'); 
title('Reconstructed image','fontsize',18)
% Compute the weights of the eigenfaces in the new image
InImWeight = []; 
for i=1:size(u,2)
    t = u(:,i)';
    WeightOfInputImage = dot(t,Difference'); 
    InImWeight = [InImWeight; WeightOfInputImage];
end

% Find distance
ll = 1:M;
figure(6)
subplot(1,2,1)
stem(ll,InImWeight)
title('Weight of Input Face','fontsize',14)

e=[];
for i=1:size(omega,2)
    q = omega(:,i);
    DiffWeight = InImWeight-q; 
    mag = norm(DiffWeight);
    e = [e, mag];
end
kk = 1:size(e,2); 
subplot(1,2,2)
stem(kk,e)
title('Eucledian distance of input image','fontsize',14)
MaximumValue=max(e);
MinimumValue=min(e);
