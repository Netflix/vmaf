test_inc = include_directories('.')

test_picture = executable('test_picture',
    ['test.c', 'test_picture.c', '../src/picture.c'],
    include_directories : [libvmaf_inc, test_inc, src_dir],
)

test('test_picture', test_picture)
