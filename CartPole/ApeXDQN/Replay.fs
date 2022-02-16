namespace CartPole.ApeXDQN
open CartPole.Core 
open DiffSharp 
open DiffSharp.Model 
open DiffSharp.Compose
open DiffSharp.Util
open System.Text
  
type Transitions = { 
    states       : Tensor[]
    actions      : Tensor[]
    rewards      : Tensor[]  
    nextStates   : Tensor[]
    isDones      : Tensor[]
} 
type SegmentTree(capacity: int) = 
   let tree = Array.zeroCreate<float> (capacity * 2)  
   
   let update (i: int, v) =  
       tree[capacity+i] <- v
       let rec loop(id: int) =  
           if id >= 1 then
               tree[id] <- 
                   let l'id = id * 2 
                   let r'id = id * 2 + 1
                   tree[l'id] + tree[r'id]   
               loop(id / 2)
       loop((capacity+i)/2)

   member this.Item  
       with get (i:int) = tree[capacity+i]
       and set i v = update(i, v) |> ignore
   
   member this.Sample() = 
        let z = Random.Uniform() * tree[1] 
        let rec loop i z = 
            if i < capacity then 
                let l = 2*i
                let r = 2*i+1
                if z > tree[l] then 
                    loop r (z-tree[l])
                else
                    loop l z
            else
                i - capacity
        loop 1 z 

    member _.Sum() = tree[1]

   override _.ToString() = 
       let sb = StringBuilder()
       for i in 0..tree.Length-2 do 
           sb.Append(sprintf "%A," tree[i]) |> ignore 
       sb.ToString() 

type Replay(bufferSize: int) = 
    let alpha = 0.6
    let beta = 0.4
    let states = Array.zeroCreate bufferSize// ResizeArray<Tensor>()
    let actions =Array.zeroCreate bufferSize
    let rewards  =Array.zeroCreate bufferSize
    let nextStates =Array.zeroCreate bufferSize
    let isDones  =Array.zeroCreate bufferSize
    let tdErrors =Array.zeroCreate bufferSize
    let priorities = SegmentTree(bufferSize) 
    let mutable idx = 0
    let mutable isFull = false 
    member _.Add(tdErrors: float[], obss: Tensor[], acts: Tensor[], nxtObss: Tensor[], rwrds: Tensor[], dones: Tensor[]) = 
        let prios = tdErrors |> Array.map(fun t -> (abs t + 0.001)** alpha) // ((dsharp.tensor tdErrors |> dsharp.abs) + 0.001) ** alpha
        for i in 0..tdErrors.Length-1 do
            let j = (idx+i)%bufferSize
            priorities[j] <- prios[i]
            states[j] <- obss[i]
            actions[j] <- acts[i]
            nextStates[j] <- nxtObss[i]
            rewards[j] <- rwrds[i]
            isDones[j] <- dones[i] 
        isFull <- isFull || (idx + tdErrors.Length) >= bufferSize
        idx <- (idx + tdErrors.Length) % bufferSize 

    member _.UpdatePriority(sampledIndices: int[], tdErrors:float[]) =  
        for i in 0..sampledIndices.Length-1 do
            let prio = (abs tdErrors[i] + 0.001) ** alpha
            priorities[sampledIndices[i]] <- prio ** alpha   

    member _.SampleMinibatch(batchSize:int) = 
         
        let sampledIndices = Array.init batchSize (fun _ -> priorities.Sample())
        let curSize = if isFull then bufferSize else idx
        let weights = 
            let sum = priorities.Sum()
            sampledIndices
            |> Array.map(fun i -> 
                let prob = priorities[i] / sum
                let weight = (prob * float curSize) ** (-beta)
                weight
            )
        let s = [| for idx in sampledIndices -> states[idx]  |]
        let a = [| for idx in sampledIndices -> actions[idx]  |]
        let r = [| for idx in sampledIndices -> rewards[idx]  |]
        let n = [| for idx in sampledIndices -> nextStates[idx]  |]
        let d = [| for idx in sampledIndices -> isDones[idx]  |]
        
        sampledIndices, weights,  { 
            states=s
            actions=a
            rewards=r  
            nextStates=n
            isDones=d
        }       