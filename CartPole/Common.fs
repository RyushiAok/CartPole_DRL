[<AutoOpen>]
module Common

open DiffSharp  

type Tensor with 

    // https://github.com/DiffSharp/DiffSharp/blob/d7ff76aa0062da56b7b4a40ca7f9be5a50dc6ffd/src/DiffSharp.Core/Tensor.fs
    // https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/smooth_l1_loss.py
    // https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html

    member input.smoothL1Loss (target:Tensor, ?reduction:string, ?beta:float) = 
        if input.shape <> target.shape then failwithf "Expecting input.shape (%A) and target.shape (%A) to be the same" input.shape target.shape
        let reduction = defaultArg reduction "mean"
        let beta = defaultArg beta 1.0
        if not (reduction = "mean" || reduction = "sum") then failwithf "Expecting reduction (%A) to be one of (mean, sum)" reduction
        let l = 
            if beta < 1e-5 then
                (input-target).abs()
            else
                let z = (input - target).abs()
                let mask1 = z.le(dsharp.tensor beta).cast(input.dtype)
                let mask2 = z.gt(dsharp.scalar beta).cast(input.dtype)  
                mask1 * (0.5 * z * z / beta) + mask2 * (z - 0.5 * beta) 
                 
        if reduction = "mean" then 
            l.mean()
        else  
            l.sum()
             

type dsharp with 

    static member smoothL1Loss  (target: Tensor) = fun (input:Tensor) -> input.smoothL1Loss(target)  

    static member smoothL1Loss  (input:Tensor, target: Tensor, ?reduction:string, ?beta:float) = input.smoothL1Loss(target,?reduction=reduction,?beta=beta) 

 
