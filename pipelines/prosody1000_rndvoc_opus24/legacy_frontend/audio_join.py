import numpy as np
SR=24000

def quiet_edge(wave,side,threshold):
    count=0;hop=120
    value=wave if side=='start' else wave[::-1]
    while count+hop<=min(len(wave),2*SR):
        if float(np.sqrt(np.mean(value[count:count+hop]**2)))>threshold:break
        count+=hop
    return count


def connect(pieces,chunks):
    pieces=[x.copy() for x in pieces]
    joins=[]
    for i in range(len(pieces)-1):
        a,b=pieces[i],pieces[i+1]
        threshold=max(.0003,min(.0015,float(np.sqrt(np.mean(np.concatenate((a,b))**2)))*.025))
        left=quiet_edge(a,'end',threshold);right=quiet_edge(b,'start',threshold)
        ending=chunks[i]['text'].rstrip(' \n”’\"」』）)')[-1:]
        gap=.24 if ending in '。.!?！？；;' else .12 if ending in '，,、：:' else .06
        removable_left=max(0,left-round(.02*SR));removable_right=max(0,right-round(.02*SR))
        remove=min(max(0,left+right-round(gap*SR)),removable_left+removable_right)
        cut_left=min(removable_left,round(remove*removable_left/max(1,removable_left+removable_right)))
        cut_right=min(removable_right,remove-cut_left)
        cut_left+=min(removable_left-cut_left,remove-cut_left-cut_right)
        if cut_left:pieces[i]=a[:-cut_left]
        if cut_right:pieces[i+1]=b[cut_right:]
        joins.append(dict(after_segment=i,ending=ending,original_quiet_ms=(left+right)/24,
                          removed_left_ms=cut_left/24,removed_right_ms=cut_right/24,
                          final_quiet_ms=(left+right-cut_left-cut_right)/24,target_max_ms=1000*gap,
                          threshold=threshold,added_silence_samples=0))
    for value in pieces:
        fade=min(72,len(value)//2)
        value[:fade]*=np.linspace(0,1,fade,dtype=np.float32)
        value[-fade:]*=np.linspace(1,0,fade,dtype=np.float32)
    return np.concatenate(pieces),joins

