% activity_filename="/home/luciano/Music/D1RS3ID2948P3.txt"
% JM_label_filename="data/labels_JMs/D1RS3ID2948P3_JM.txt"
% temp_wav = "temporal.wav";
function [mi_grazing,mi_rumination,snr_grazing,snr_rumination] = estimate_snr_mi_inward(activity_filename,JM_label_filename,temp_wav)

Fs=44100;
[Label_begin,Label_end,Labels_activity]=import_labels(activity_filename);
mi_signal_grazing=[];
mi_signal_rumination=[];
snr_signal_grazing=[];
snr_signal_rumination=[];
info = audioinfo(temp_wav);
for j=1:numel(Labels_activity)
    if (strcmp(Labels_activity(j),"Grazing") || strcmp(Labels_activity(j),"Rumination") || strcmp(Labels_activity(j),"Rumination (lying-down)") || ...
            strcmp(Labels_activity(j),"Rumination (windy)") || strcmp(Labels_activity(j),"Rumination (raining)"))
        intervalo = [floor(Label_begin(j)*Fs+1), floor(Label_end(j)*Fs)];
        
        if intervalo(2)>info.TotalSamples
            intervalo(2)=info.TotalSamples;
        end
        if Label_end(j)>info.TotalSamples/Fs
            Label_end(j)=info.TotalSamples/Fs;
        end
        [x,~]=audioread(temp_wav,intervalo);
        [mi_value,snr_value] = compute_snr_mi(x,JM_label_filename,Label_begin(j),Label_end(j));
        if strcmp(Labels_activity(j),"Grazing")
            mi_signal_grazing = [mi_signal_grazing , mi_value];
            snr_signal_grazing = [snr_signal_grazing , snr_value];
        else
            mi_signal_rumination = [mi_signal_rumination , mi_value];
            snr_signal_rumination = [snr_signal_rumination , snr_value];
        end
    end
end

mi_grazing = mean(mi_signal_grazing,'omitnan');
mi_rumination = mean(mi_signal_rumination,'omitnan');
snr_grazing = mean(snr_signal_grazing,'omitnan');
snr_rumination = mean(snr_signal_rumination,'omitnan');

end

function [Inicio,Fin,Label] = import_labels(filename, startRow, endRow)
%% Initialize variables.
delimiter = '\t';
if nargin<=2
    startRow = 1;
    endRow = inf;
end

%% Format for each line of text:
%   column1: double (%f)
%	column2: double (%f)
%   column3: text (%s)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%f%f%s%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow(1)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow(block)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    for col=1:length(dataArray)
        dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    end
end

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Allocate imported array to column variable names
Inicio = dataArray{:, 1};
Fin = dataArray{:, 2};
Label = dataArray{:, 3};
end


function  [result_MI,result_SNR] = compute_snr_mi(audiosignal,event_file,Label_begin,Label_end)
%calcula el "modulation index" y "SNR"
    [full_labels_start,full_labels_end]=import_labels(event_file);
    
    etiquetas_inicio =  full_labels_start((full_labels_start >= Label_begin) & (full_labels_end <= Label_end));
    etiquetas_fin =  full_labels_end((full_labels_start >= Label_begin) & (full_labels_end <= Label_end));
    etiquetas_inicio=etiquetas_inicio - Label_begin;
    etiquetas_fin = etiquetas_fin - Label_begin;
    MIi=zeros(size(etiquetas_inicio,1),1);
    SNRi=zeros(size(etiquetas_inicio,1),1);
    Fs=44100;
    etiquetas_inter(:,1)=round((etiquetas_inicio)*Fs);
    etiquetas_inter(:,2)=round((etiquetas_fin)*Fs);
    
%     if(etiquetas_inter(:,2)>numel(audiosignal))
%         etiquetas_inter(:,2)=numel(audiosignal);
%     end
    
    for i=2:size(etiquetas_inicio,1)
        evento=audiosignal(etiquetas_inter(i,1):etiquetas_inter(i,2));
        inter_evento=audiosignal(etiquetas_inter(i-1,2)+1:etiquetas_inter(i,1)-1);
        
        %compute Modulation Index
        MIi(i)=abs(rms(evento)-rms(inter_evento))/(rms(evento)+rms(inter_evento));
        %compute SNR
        if numel(inter_evento) > numel(evento)
            inter_evento=inter_evento(1:numel(evento));
        end    
        senial = [ inter_evento ; evento ]; %pego la parte del ruido a la señal
        IS=numel(inter_evento)/Fs;
        try
            signal_clean=SSMultibandKamath02(senial,Fs,IS);
            signal_clean = signal_clean(numel(inter_evento+1): end); %elimino la parte del ruido y dejo solo la señal
            signal=evento(1:numel(signal_clean));
            signal_noise=signal-signal_clean;
            SNRi(i)=20*log10(rms(signal_clean)/rms(signal_noise));
        catch
            SNRi(i)=nan;
        end
    end
    
result_MI = mean(MIi(2:end),'omitnan');
result_SNR = mean(SNRi(2:end),'omitnan');
end


function output=SSMultibandKamath02(signal,fs,IS)

% OUTPUT=SSMULTIBANDKAMATH02(S,FS,IS)
% Multi-band Spectral subtraction [Kamath2002]
% subtraction with adjusting subtraction factor. the adjustment is
% according to local a postriori SNR and the frequency band.
% S is the noisy signal, FS is the sampling frequency and IS is the initial
% silence (noise only) length in seconds (default value is .25 sec)
%
% April-04
% Esfandiar Zavarehei

if (nargin<3 | isstruct(IS))
    IS=.25; %seconds
end
% W=fix(.025*fs); %Window length is 25 ms
W=fix(.04*fs); %Window length is 40 ms
nfft=W;
SP=.4; %Shift percentage is 40% (10ms) %Overlap-Add method works good with this value(.4)
wnd=hamming(W);

% IGNORE THIS SECTION FOR CAMPATIBALITY WITH ANOTHER PROGRAM FROM HERE.....
if (nargin>=3 & isstruct(IS))%This option is for compatibility with another programme
    W=IS.windowsize
    SP=IS.shiftsize/W;
    nfft=IS.nfft;
    wnd=IS.window;
    if isfield(IS,'IS')
        IS=IS.IS;
    else
        IS=.25;
    end
end
% .......IGNORE THIS SECTION FOR CAMPATIBALITY WITH ANOTHER PROGRAM T0 HERE

NIS=fix((IS*fs-W)/(SP*W) +1);%number of initial silence segments
Gamma=2;%Magnitude Power (1 for magnitude spectral subtraction 2 for power spectrum subtraction)

y=segment(signal,W,SP,wnd);
Y=fft(y,nfft);
YPhase=angle(Y(1:fix(end/2)+1,:)); %Noisy Speech Phase
Y=abs(Y(1:fix(end/2)+1,:)).^Gamma;%Specrogram
numberOfFrames=size(Y,2);
FreqResol=size(Y,1);

N=mean(Y(:,1:NIS)')'; %initial Noise Power Spectrum mean

NoiseCounter=0;
NoiseLength=9;%This is a smoothing factor for the noise updating

Beta=.03;
minalpha=1;
maxalpha=5;
% maxalpha=3;
minSNR=-5;
% maxSNR=20;
maxSNR=15;
alphaSlope=(minalpha-maxalpha)/(maxSNR-minSNR);
alphaShift=maxalpha-alphaSlope*minSNR;

BN=Beta*N;

%Delta is a frequency dependent coefficient
% Delta=1.5*ones(size(BN));
% Delta(1:fix((-2000+fs/2)*FreqResol*2/fs))=2.5; %if the frequency is lower than FS/2 - 2KHz
% Delta(1:fix(1000*FreqResol*2/fs))=1; %if the frequency is lower than 1KHz

% Delta=4.0*ones(size(BN));
% Delta(1:fix((-3000+fs/2)*FreqResol*2/fs))=2.5; %if the frequency is lower than FS/2 - 2KHz
% Delta(1:fix(2000*FreqResol*2/fs))=1; %if the frequency is lower than 1KHz

Delta=8.0*ones(size(BN));
Delta(1:fix(8000*FreqResol*2/fs))=4; %if the frequency is lower than 8KHz
Delta(1:fix(4000*FreqResol*2/fs))=2; %if the frequency is lower than 4KHz
Delta(1:fix(2500*FreqResol*2/fs))=1.25; %if the frequency is lower than KHz
Delta(1:fix(1500*FreqResol*2/fs))=1; %if the frequency is lower than 1.5KHz
Delta(1:fix(100*FreqResol*2/fs))=2; %if the frequency is lower than 100Hz


for i=1:numberOfFrames
    [NoiseFlag, SpeechFlag, NoiseCounter, Dist]=vad(Y(:,i).^(1/Gamma),N.^(1/Gamma),NoiseCounter); %Magnitude Spectrum Distance VAD
    if SpeechFlag==0
        N=(NoiseLength*N+Y(:,i))/(NoiseLength+1); %Update and smooth noise
        BN=Beta*N;
    end
    
    SNR=10*log(Y(:,i)./N);
    alpha=alphaSlope*SNR+alphaShift;
    alpha=max(min(alpha,maxalpha),minalpha);
    
    D=Y(:,i)-(Delta.*alpha.*N); %Nonlinear (Non-uniform) Power Specrum Subtraction
%     try
        X(:,i)=max(D,BN); %if BY>D X=BY else X=D which sets very small values of subtraction result to an attenuated 
                      %version of the input power spectrum.
%     catch
%         pause
%     end
end

output=OverlapAdd2(X.^(1/Gamma),YPhase,W,SP*W);
end

function Seg=segment(signal,W,SP,Window)

% SEGMENT chops a signal to overlapping windowed segments
% A= SEGMENT(X,W,SP,WIN) returns a matrix which its columns are segmented
% and windowed frames of the input one dimentional signal, X. W is the
% number of samples per window, default value W=256. SP is the shift
% percentage, default value SP=0.4. WIN is the window that is multiplied by
% each segment and its length should be W. the default window is hamming
% window.
% 06-Sep-04
% Esfandiar Zavarehei

if nargin<3
    SP=.4;
end
if nargin<2
    W=256;
end
if nargin<4
    Window=hamming(W);
end
Window=Window(:); %make it a column vector

L=length(signal);
SP=fix(W.*SP);
N=fix((L-W)/SP +1); %number of segments

Index=(repmat(1:W,N,1)+repmat((0:(N-1))'*SP,1,W))';
hw=repmat(Window,1,N);
Seg=signal(Index).*hw;
end

function [NoiseFlag, SpeechFlag, NoiseCounter, Dist]=vad(signal,noise,NoiseCounter,NoiseMargin,Hangover)

%[NOISEFLAG, SPEECHFLAG, NOISECOUNTER, DIST]=vad(SIGNAL,NOISE,NOISECOUNTER,NOISEMARGIN,HANGOVER)
%Spectral Distance Voice Activity Detector
%SIGNAL is the the current frames magnitude spectrum which is to labeld as
%noise or speech, NOISE is noise magnitude spectrum template (estimation),
%NOISECOUNTER is the number of imediate previous noise frames, NOISEMARGIN
%(default 3)is the spectral distance threshold. HANGOVER ( default 8 )is
%the number of noise segments after which the SPEECHFLAG is reset (goes to
%zero). NOISEFLAG is set to one if the the segment is labeld as noise
%NOISECOUNTER returns the number of previous noise segments, this value is
%reset (to zero) whenever a speech segment is detected. DIST is the
%spectral distance. 
%Saeed Vaseghi
%edited by Esfandiar Zavarehei
%Sep-04

if nargin<4
    NoiseMargin=3;
%     NoiseMargin=5;
end
if nargin<5
    Hangover=8;
end
if nargin<3
    NoiseCounter=0;
end
    
FreqResol=length(signal);

SpectralDist= 20*(log10(signal)-log10(noise));
SpectralDist(find(SpectralDist<0))=0;

Dist=mean(SpectralDist); 
if (Dist < NoiseMargin) 
    NoiseFlag=1; 
    NoiseCounter=NoiseCounter+1;
else
    NoiseFlag=0;
    NoiseCounter=0;
end

% Detect noise only periods and attenuate the signal     
if (NoiseCounter > Hangover) 
    SpeechFlag=0;    
else 
    SpeechFlag=1; 
end 
end

function ReconstructedSignal=OverlapAdd2(XNEW,yphase,windowLen,ShiftLen)

%Y=OverlapAdd(X,A,W,S);
%Y is the signal reconstructed signal from its spectrogram. X is a matrix
%with each column being the fft of a segment of signal. A is the phase
%angle of the spectrum which should have the same dimension as X. if it is
%not given the phase angle of X is used which in the case of real values is
%zero (assuming that its the magnitude). W is the window length of time
%domain segments if not given the length is assumed to be twice as long as
%fft window length. S is the shift length of the segmentation process ( for
%example in the case of non overlapping signals it is equal to W and in the
%case of %50 overlap is equal to W/2. if not givven W/2 is used. Y is the
%reconstructed time domain signal.
%Sep-04
%Esfandiar Zavarehei

if nargin<2
    yphase=angle(XNEW);
end
if nargin<3
    windowLen=size(XNEW,1)*2;
end
if nargin<4
    ShiftLen=windowLen/2;
end
if fix(ShiftLen)~=ShiftLen
    ShiftLen=fix(ShiftLen);
%     disp('The shift length have to be an integer as it is the number of samples.')
%     disp(['shift length is fixed to ' num2str(ShiftLen)])
end

[FreqRes FrameNum]=size(XNEW);

Spec=XNEW.*exp(j*yphase);

if mod(windowLen,2) %if FreqResol is odd
    Spec=[Spec;flipud(conj(Spec(2:end,:)))];
else
    Spec=[Spec;flipud(conj(Spec(2:end-1,:)))];
end
sig=zeros((FrameNum-1)*ShiftLen+windowLen,1);
weight=sig;
for i=1:FrameNum
    start=(i-1)*ShiftLen+1;
    spec=Spec(:,i);
    sig(start:start+windowLen-1)=sig(start:start+windowLen-1)+real(ifft(spec,windowLen));
end
ReconstructedSignal=sig;
end
