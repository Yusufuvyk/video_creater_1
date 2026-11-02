% The same variables saved as MATLAB v7 and v7.3, for comparing GIOVA's two readers.
out = fileparts(mfilename('fullpath'));

eeg = reshape(1:3000, 3, 1000) / 7;            % channels x samples
fs = 250;
srate_int = int16(512);
labels = {'C3', 'Cz', 'C4'};
name = 'Subject ölçüm ÇĞİÖŞÜ';
charmat = ['ab'; 'cd'];
flags = logical([1 0 1]);
empty_num = zeros(0, 3);
empty_cell = {};
empty_char = '';
cplx = [1+2i, 3-4i];
single_arr = single(magic(4));
vec_col = (1:5)';
vec_row = 1:5;
nd = reshape(1:24, 2, 3, 4);
uint8_arr = uint8([0 128 255]);

s = struct();
s.X = reshape(1:60, 20, 3);                    % samples x channels
s.fs = 250;
s.classes = {'left', 'right'};
s.trial = [1; 11];
s.y = [1; 2];
s.sub = struct('inner', 5, 'text', 'deep');

runs = cell(1, 3);
for k = 1:3
    r = struct();
    r.X = reshape((1:400) + 1000 * k, 100, 4);
    if k == 1
        r.trial = [];
        r.y = [];
        r.artifacts = [];
    else
        r.trial = [1; 51];
        r.y = [1; 2];
        r.artifacts = [0; 1];
    end
    r.fs = 100;
    r.classes = {'a', 'b'};
    r.gender = 'f';
    r.age = 22;
    runs{k} = r;
end
data = runs;

chanlocs = struct('labels', {'C3', 'Cz', 'C4'}, 'X', {1.0, 2.0, 3.0});

EEG = struct();
EEG.data = single(reshape(1:800, 4, 200));
EEG.srate = 100;
EEG.event = struct('latency', {10, 120}, 'type', {'S 1', 'S 2'});

grid_struct = struct('v', {11, 12; 21, 22}, 'name', {'a11', 'a12'; 'a21', 'a22'});   % 2 x 2
grid_cell = {11, 12, 13; 21, 22, 23};                                               % 2 x 3
col_cell = {1; 2; 3};
wide = reshape(1:12, 3, 4) + 0.5;                                                   % 3 x 4, not square
% trials already cut: channels x samples x trials, as EEGLAB keeps epochs
epo = reshape(1:(4 * 50 * 6), 4, 50, 6) / 10;
epo_y = [1 2 1 2 1 2];
epo_classes = {'left', 'right'};
epo_fs = 100;
epo_bad = [0 0 1 0 0 0];
% MATLAB objects GIOVA does not read: named, not dropped in silence
words = ["alpha", "beta"];
tbl = table([1; 2], {'a'; 'b'});
cellmix = {1, 'two', [3 4 5]};
nested_cell = {{1, 2}, {3}};
one_cell = {7};

names = {'eeg', 'fs', 'srate_int', 'labels', 'name', 'charmat', 'flags', 'empty_num', 'empty_cell', ...
         'empty_char', 'cplx', 'single_arr', 'vec_col', 'vec_row', 'nd', 'uint8_arr', 's', 'data', ...
         'chanlocs', 'EEG', 'cellmix', 'nested_cell', 'one_cell', 'grid_struct', 'grid_cell', 'col_cell', 'wide', ...
         'epo', 'epo_y', 'epo_classes', 'epo_fs', 'epo_bad', 'words', 'tbl'};
save(fullfile(out, 'fixture_v7.mat'), names{:}, '-v7');
save(fullfile(out, 'fixture_v73.mat'), names{:}, '-v7.3');
disp('saved');
