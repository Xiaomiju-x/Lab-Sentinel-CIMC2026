"""
gd32ai_inspect_tflite.py — parse the GD32-AI-Tool-produced .tflite FLATBUFFER and
dump its op graph + tensor shapes + weight buffers. No TensorFlow needed (pure
flatbuffer schema). Confirms exactly what the tool emitted so we can deploy THAT
on-chip (weights extracted from the tool's file, not re-derived from PyTorch).

Run: cd CIMC/model && python gd32ai_inspect_tflite.py gd32ai_export/ai4_fusion_gd32ai.tflite
"""
import sys
import numpy as np
from tflite.Model import Model
from tflite.BuiltinOperator import BuiltinOperator
from tflite.TensorType import TensorType

_BO = {v: k for k, v in BuiltinOperator.__dict__.items() if isinstance(v, int)}
_TT = {v: k for k, v in TensorType.__dict__.items() if isinstance(v, int)}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "gd32ai_export/ai4_fusion_gd32ai.tflite"
    buf = open(path, "rb").read()
    m = Model.GetRootAs(buf, 0)
    print(f"file={path}  version={m.Version()}  subgraphs={m.SubgraphsLength()}")
    sg = m.Subgraphs(0)
    print(f"tensors={sg.TensorsLength()} ops={sg.OperatorsLength()} "
          f"inputs={list(sg.InputsAsNumpy())} outputs={list(sg.OutputsAsNumpy())}\n")

    def tshape(t):
        return list(sg.Tensors(t).ShapeAsNumpy()) if sg.Tensors(t).ShapeLength() else []

    def ttype(t):
        return _TT.get(sg.Tensors(t).Type(), "?")

    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        code = m.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        ins = [op.Inputs(j) for j in range(op.InputsLength())]
        outs = [op.Outputs(j) for j in range(op.OutputsLength())]
        print(f"op[{i}] {_BO.get(code, code)}")
        for t in ins:
            buf_i = sg.Tensors(t).Buffer()
            b = m.Buffers(buf_i)
            n = b.DataLength()
            print(f"    in  t{t:2d} {ttype(t):7s} shape={tshape(t)} buf={buf_i} bytes={n}")
        for t in outs:
            print(f"    out t{t:2d} {ttype(t):7s} shape={tshape(t)}")


if __name__ == "__main__":
    main()
