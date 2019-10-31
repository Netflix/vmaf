# Revision file (vcs_version.h) generation
vmaf_git_dir = join_paths(libvmaf_src_root + '/../', '.git')
rev_target = vcs_tag(command: [
        'git', '--git-dir', vmaf_git_dir,
        'describe', '--tags', '--long',
        '--match', '?.*.*', '--always'
    ],
    input: 'vcs_version.h.in',
    output: 'vcs_version.h'
)

subdir('libvmaf')
