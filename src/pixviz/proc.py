from pathlib import Path

import cv2

__all__ = ['rotate_video']


def rotate_video(input_path: Path | str,
                 output_path: Path | str,
                 angle: float, *,
                 fourcc_type: str = 'MJPG') -> None:
    """
    Rotate the video

    :param input_path: input video path
    :param output_path: rotated output path
    :param angle: rotation angle in degree
    :param fourcc_type: codec type
    """
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*fourcc_type)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print('processing...')

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Get the center of the frame
        center = (width // 2, height // 2)

        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_frame = cv2.warpAffine(frame, rot_matrix, (width, height))

        out.write(rotated_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print('processing finished!')


def main():
    import argparse

    ap = argparse.ArgumentParser()

    ap.add_argument(metavar='FILE', type=str, dest='input', help='input video file')
    ap.add_argument(metavar='FILE', type=str, dest='output', help='output video file')
    ap.add_argument('-A', '--angle', type=float, dest='angle', help='rotation_angle')
    ap.add_argument('--code', type=str, default='MJPG', dest='codec',
                    help='four bytes used to uniquely identify data formats')

    opt = ap.parse_args()

    rotate_video(opt.input, opt.output, opt.angle, fourcc_type=opt.codec)


if __name__ == '__main__':
    main()
