from app.routes.common_router_functions import *



router = APIRouter()


class OneHotDefs(BaseModel):
    OneHotDefs : Dict 



@router.post("/onehotencoding")
async def one_hotenconding(data: Data,
                           column_name):
    
    # column_defs = column_defs.OneHotDefs
    # print(column_defs)

    df = convert_to_df(data)

    df = funs.onehotEncoding(df, column_name)


    return {"message": "Data received", "data": df.to_json(orient='records')}



@router.post("/scale")
async def scale_column(data: Data,
                           column_name, method, new_min:int=0, new_max:int=1):
    
    # column_defs = column_defs.OneHotDefs
    # print(column_defs)
    df = convert_to_df(data)

    if method =="normalize":
        df = funs.normalize_column(df, column_name,new_min=new_min,new_max=new_max)
    elif method =="standardize":
        pass
    else:
        raise(f" {method} not a valid method")

    return {"message": "Data received", "data": df.to_json(orient='records')}





@router.get("/test")
async def test(k):
    print(k)
    return k
    
