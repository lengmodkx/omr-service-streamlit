"""gRPC 调用示例"""
import sys

import grpc

sys.path.insert(0, "../..")

from omr_service.rpc import omr_pb2, omr_pb2_grpc


def main():
    channel = grpc.insecure_channel("127.0.0.1:20884")
    stub = omr_pb2_grpc.OmrServiceStub(channel)

    # 1. 解析黄金模板
    columns = [
        omr_pb2.ColumnConfig(
            x1=50, y1=50, x2=550, y2=350,
            start_q=1, num_q=3, num_options=4,
        )
    ]
    tpl_req = omr_pb2.GoldenTemplateRequest(
        template_id=1001,
        template_image_url="https://example.com/template.jpg",
        columns=columns,
    )
    tpl_resp = stub.ParseGoldenTemplate(tpl_req)
    print("ParseGoldenTemplate:", tpl_resp.code, tpl_resp.message)

    # 2. 识别答题卡
    rec_req = omr_pb2.RecognizeRequest(
        template_id=1001,
        scan_image_url="https://example.com/scan.jpg",
    )
    rec_resp = stub.RecognizeByTemplate(rec_req)
    print("RecognizeByTemplate:", rec_resp.code, rec_resp.message)
    for a in rec_resp.answers:
        print(f"  Q{a.q}: {a.answer} ({a.status})")


if __name__ == "__main__":
    main()
