from scannerpy import Database, Job, BulkJob, ColumnType, DeviceType
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) <= 1:
    print('Usage: {:s} path/to/your/video/file.mp4'.format(sys.argv[0]))
    sys.exit(1)

movie_path = sys.argv[1]
print('Detecting objects in movie {}'.format(movie_path))
movie_name = os.path.splitext(os.path.basename(movie_path))[0]


with Database() as db:
    [input_table], failed = db.ingest_videos([('example', movie_path)], force=True)

    print(db.summarize())

    db.register_op('ObjDetect', [('frame', ColumnType.Video)], ['out_frame'])
    kernel_path = script_dir + '/obj_detect_kernel.py'
    db.register_python_kernel('ObjDetect', DeviceType.CPU, kernel_path)

    frame = db.ops.FrameInput()
    out_frame = db.ops.ObjDetect(frame = frame)
    output_op = db.ops.Output(columns=[out_frame])

    job = Job(op_args={
        frame: db.table('example').column('frame'),
        output_op: 'example_obj_detect'
    })
    bulk_job = BulkJob(output=output_op, jobs=[job])
    [out_table] = db.run(bulk_job, force=True, pipeline_instances_per_node=1)
    out_table.column('out_frame').save_mp4(movie_name + '_obj_detect')

    print('Successfully generated {:s}_obj_detect.mp4'.format(movie_name))