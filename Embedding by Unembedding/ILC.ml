module ILC = struct 
  module FromFst = struct 
    type _ t = 
    | FFst : 'a -> ('a * 'd) t  
  end 

  module FromSnd = struct 
    type _ t = 
    | FSnd : 'd -> ('a * 'd) t  
  end 


  module H = HList(FromFst)
  module D = HList(FromSnd)

  open FromSnd
  open FromFst
  open H 
  open D 

  type ('x, 'xs) ilc = ('xs H.hlist -> 'x FromFst.t) * ('xs H.hlist -> 'xs D.hlist -> 'x FromSnd.t)

  let varN : type x xs. (x * xs) H.hlist -> x FromFst.t = 
    function (HCons(FFst x,_)) -> FFst x  

  let varD : type x xs. (x * xs) H.hlist -> (x * xs) D.hlist -> x FromSnd.t =
    fun _ -> function (HCons(FSnd d,_)) -> FSnd d 

  module VarILC = Variables (
    struct 
      type ('x, 'xs) sem = ('x, 'xs) ilc 
      let var : type x xs. (x, (x * xs)) ilc = (varN, varD) 
      let weaken : type x y xs. (x, xs) ilc -> (x, (y * xs)) ilc = fun (f,df) -> 
        (function (HCons(_,env)) -> f env), 
        (function (HCons(_,env)) -> function (HCons(_,denv)) -> df env denv) 
    end 
  )
end

module SemILC = struct
  open ILC

  let addSem : type env. (int * int, env) ilc -> (int * int, env) ilc -> (int * int, env) ilc =
    fun (f, df) (g, dg) -> 
      (fun theta -> let FFst v = f theta in let FFst w = g theta in FFst (v + w)) , 
      (fun theta dtheta -> let FSnd dv = df theta dtheta in let FSnd dw = dg theta dtheta in FSnd (dv + dw))

  let mulSem : type env. 
        (int * int, env) ilc -> (int * int, env) ilc -> (int * int, env) ilc
      = fun (f, df) (g, dg) -> 
          (fun theta -> 
            let FFst v = f theta in 
            let FFst w = g theta in 
            FFst (v * w)),
          (fun theta dtheta -> 
            let FFst v = f theta in 
            let FFst w = g theta in
            let FSnd dv = df theta dtheta in 
            let FSnd dw = dg theta dtheta in 
            FSnd (v * dw + dv * w + dv * dw) 
            )

  let fstSem : type env a da b db. ((a * b) * (da * db) , env) ilc -> (a * da, env) ilc 
        = fun (f, df) -> 
          (fun theta -> 
            let FFst (v, _) = f theta in 
              FFst v), 
          (fun theta dtheta -> 
            let FSnd (dv,_) = df theta dtheta in
              FSnd dv  
            )

  let sndSem : type env a da b db. ((a * b) * (da * db) , env) ilc -> (b * db, env) ilc 
        = fun (f, df) -> 
          (fun theta -> 
            let FFst (_, v) = f theta in 
              FFst v), 
          (fun theta dtheta -> 
            let FSnd (_, dv) = df theta dtheta in
              FSnd dv  
            )
    
  let letSem : type env a b. 
        (a, env) ilc -> (b, (a * env)) ilc -> (b, env) ilc 
        = fun (f, df) (g, dg) -> 
            (fun theta -> 
              let FFst v = f theta in 
              let FFst w = g (HCons(FFst v,theta)) in 
                FFst w), 
            (fun theta dtheta -> 
              let FFst v = f theta in 
              let FSnd dv = df theta dtheta in 
                dg (HCons(FFst v, theta)) (HCons(FSnd dv,dtheta))
              )
end 

module type HILC = sig 
  type 'a expr 

  val add : (int * int) expr -> (int * int) expr -> (int * int) expr 
  val mul : (int * int) expr -> (int * int) expr -> (int * int) expr
  val ( let* ) : 'a expr -> ('a expr -> 'b expr) -> 'b expr  
  val fst : (('a * 'b) * ('da * 'db)) expr -> ('a * 'da) expr 
  val snd : (('a * 'b) * ('da * 'db)) expr -> ('b * 'db) expr 


  val runOpen : (('a * 'da) expr -> ('b * 'db) expr) -> ('a -> 'b) * ('a -> 'da -> 'db)

end 

module HILC_sem : HILC with type 'a expr = 'a ILC.VarILC.envi = struct 
  open ILC open VarILC
  type 'a expr = 'a envi 

  let add = fun e1 e2 -> 
    let argSpec = XCons (XZ, XCons (XZ, XNil)) in 
      ILC.VarILC.toF argSpec (liftSO' {f = fun x -> fromF argSpec SemILC.addSem x}) e1 e2 

  let mul = fun e1 e2 -> 
    let argSpec = XCons (XZ, XCons (XZ, XNil)) in 
      toF argSpec (liftSO' {f = fun x -> fromF argSpec SemILC.mulSem x}) e1 e2

  let ( let* ) = fun e1 e2 -> 
    let argSpec = XCons (XZ, XCons (XS XZ, XNil)) in 
      toF argSpec (liftSO' {f = fun x -> fromF argSpec SemILC.letSem x}) 
    e1 e2 

  let fst = fun e -> 
    let argSpec = XCons (XZ, XNil) in 
      toF argSpec (liftSO' {f = fun x -> fromF argSpec SemILC.fstSem x}) e 

  let fst = fun e -> 
    let argSpec = XCons (XZ, XNil) in 
      toF argSpec (liftSO' {f = fun x -> fromF argSpec SemILC.SndSem x}) e 

  let runOpen : 
    type a da b db. ((a * da) expr -> (b * db) expr) -> (a -> b) * (a -> da -> db) = 
    fun f -> 
      let (g, dg) = VarILC.runOpen f 
      in (fun x -> let FFst v = g (HCons(FFst x, HNil)) in v ), 
         (fun x dx -> let FSnd dv = dg (HCons (FFst x,HNil)) (HCons (FSnd dx,HNil)) in dv) 
end
