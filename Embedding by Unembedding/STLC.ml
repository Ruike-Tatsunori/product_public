module STLC = struct
  module VEnv = HList(struct type 'a t = 'a end)

  type ('a, 'e) stlc = { runSim : 'e VEnv.hlist -> 'a }

  module VarSTLC = Variables (
    struct 
      type ('a, 'b) sem = ('a, 'b) stlc 

      let var = { runSim = function VEnv.HCons (x, _) -> x }
    
      let weaken = fun stlc -> 
        { runSim = function VEnv.HCons (_, venv') -> stlc.runSim venv' }
  end 
  )
end


  
module SemSTLC = struct
  open STLC
  
  let appSem = fun fTerm aTerm ->
    { runSim = fun venv -> 
      let f = fTerm.runSim venv in
      let x = aTerm.runSim venv in
      f x }
  
  let lamSem = fun bTerm ->
    { runSim = fun venv x ->
      let g' = VEnv.HCons (x, venv) in 
      bTerm.runSim g' }    
end

module type HSTLC = sig
  type 'a expr 
  val app : ('a -> 'b) expr -> 'a expr -> 'b expr
  val lam : ('a expr -> 'b expr) -> ('a -> 'b) expr
  val runOpenSTLC : ('a expr -> 'b expr) -> ('b, 'a * unit) stlc
end

module STLC_sem : HSTLC with type 'a expr = 'a STLC.VarSTLC.envi = struct 
  type 'a expr = 'a VarSTLC.envi
  open STLC open VarSTLC 

  let app : type a b. (a -> b) envi -> a envi -> b envi = fun e1 e2 -> 
    let argSpec = XCons (XZ, XCons (XZ, XNil)) in
      toF argSpec (liftSO' {f = fun x -> fromF argSpec SemSTLC.appSem x}) e1 e2

  let lam : type a b. (a envi -> b envi) -> (a -> b) envi = fun e ->
    let argSpec = XCons (XS XZ, XNil) in 
      toF argSpec (liftSO' {f = fun x -> fromF argSpec SemSTLC.lamSem x}) e

  let runOpenSTLC : type a b.(a expr -> b expr) -> (b, a * unit) stlc = fun f ->
    VarSTLC.runOpen f 
      
  end 

let runSTLC = fun t x ->
  let g = VEnv.HCons (x, VEnv.HNil) in 
  t.runSim g
