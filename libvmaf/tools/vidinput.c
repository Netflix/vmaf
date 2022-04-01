/*Daala video codec
Copyright (c) 2002-2013 Daala project contributors.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.*/

#include "vidinput.h"
#include <stdlib.h>
#include <string.h>

extern video_input_vtbl Y4M_INPUT_VTBL;
extern video_input_vtbl YUV_INPUT_VTBL;

int raw_input_open(video_input *_vid,FILE *_fin,
                   unsigned width, unsigned height,
                   int pix_fmt, unsigned bitdepth)
{
  void *ctx;
  if ((ctx = YUV_INPUT_VTBL.open_raw(_fin, width, height,
                                     pix_fmt, bitdepth)) != NULL)
  {
    _vid->vtbl=&YUV_INPUT_VTBL;
    _vid->ctx=ctx;
    _vid->fin=_fin;
    return 0;
  }
  else fprintf(stderr,"Unknown file type.\n");
  return -1;
}

int video_input_open(video_input *_vid,FILE *_fin) {
  void *ctx;
  if ((ctx = Y4M_INPUT_VTBL.open(_fin))!=NULL){
    _vid->vtbl=&Y4M_INPUT_VTBL;
    _vid->ctx=ctx;
    _vid->fin=_fin;
    return 0;
  }
  else fprintf(stderr,"Unknown file type.\n");
  return -1;
}

void video_input_get_info(video_input *_vid,video_input_info *_info) {
  (*_vid->vtbl->get_info)(_vid->ctx,_info);
}

int video_input_fetch_frame(video_input *_vid,
 video_input_ycbcr _ycbcr,char _tag[5]) {
  return (*_vid->vtbl->fetch_frame)(_vid->ctx,_vid->fin,_ycbcr,_tag);
}

void video_input_close(video_input *_vid) {
  (*_vid->vtbl->close)(_vid->ctx);
  free(_vid->ctx);
  fclose(_vid->fin);
}
