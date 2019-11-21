test_inc = include_directories('.')

test_picture = executable('test_picture',
    ['test.c', 'test_picture.c', '../src/picture.c', '../src/mem.c'],
    include_directories : [libvmaf_inc, test_inc, '../src/'],
)

test_feature_collector = executable('test_feature_collector',
    ['test.c', 'test_feature_collector.c',],
    include_directories : [libvmaf_inc, test_inc, '../src/feature/'],
)

test('test_picture', test_picture)
test('test_feature_collector', test_feature_collector)
